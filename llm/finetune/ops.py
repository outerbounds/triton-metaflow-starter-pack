import os
import shutil
from subprocess import check_output
from metaflow import S3, current, Parameter
import logging 
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

LLAMA_PYTHON_BACKEND_CONFIG = """
name: "{deployment_name}"
backend: "python"
max_batch_size: 8
input [
  {{
    name: "prompt"
    data_type: TYPE_STRING  
    dims: [-1]
  }}
]
output [
  {{
    name: "generated_text"
    data_type: TYPE_STRING  
    dims: [-1]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
  }}
]

dynamic_batching {{ 
  preferred_batch_size: [2, 4, 8] 
  max_queue_delay_microseconds: 3000000
}}
"""

def make_tar_bytes(source_dir):
    from tarfile import ExtractError, TarFile
    from io import BytesIO
    buf = BytesIO()
    with TarFile(mode="w", fileobj=buf) as tar:
        tar.add(source_dir)
    return buf.getvalue()

class ModelStore():

    model_repo = Parameter("model-repo", required=True, help="S3 path to model repo")

    def store_transformer(self, save_pretrained_path):

        self.deployment_name = f"{current.flow_name}-{current.run_id}"
        self.config = LLAMA_PYTHON_BACKEND_CONFIG.format(deployment_name=self.deployment_name) 

        # make file structure like this and zip it
        # | FLOW_NAME-RUN_ID
        # ----| 1
        # --------| backend.py
        # --------| setup_env.py
        # --------| <save_pretrained_path>
        # ------------| model
        # ----------------| adapter_config.json
        # ----------------| adapter_model.bin
        # ------------| tokenizer
        # ----------------| special_tokens_map.json
        # ----------------| tokenizer_config.json
        # ----------------| tokenizer.json
        # ----| config.pbtxt

        # make the directory structure
        os.makedirs(f"{self.deployment_name}/1", exist_ok=True)

        # write config
        with open(f"{self.deployment_name}/config.pbtxt", "w") as f:
            f.write(self.config)

        # run-info.txt will be used in the backend.py on the inference server
        with open(f"{self.deployment_name}/1/run-info.txt", "w") as f:
            f.write(f"flow_name={current.flow_name}\nrun_id={current.run_id}")

        # rewrite backend.py as model.py and include in pkg
        _ = shutil.move("backend.py", f"{self.deployment_name}/1/model.py")

        # write setup_env
        _ = shutil.move("setup_env.py", f"{self.deployment_name}/1/setup_env.py")

        # write model
        _ = shutil.move(save_pretrained_path, f"{self.deployment_name}/1/{save_pretrained_path}")
        
        with S3(s3root=self.model_repo) as s3:
            url = s3.put(key=self.deployment_name, obj=make_tar_bytes(self.deployment_name))
            logging.info(f"The model and its Triton config has deployed at {url}")
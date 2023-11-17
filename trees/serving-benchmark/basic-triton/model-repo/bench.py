from metaflow import S3
import logging
import os 
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

s3_root = "s3://outerbounds-datasets/triton/tree-models-benchmark/"

# pull model from s3
with S3(s3root=s3_root) as s3:
    objs = s3.get_recursive("triton")

    cwd = os.getcwd()
    dir_name = os.path.join(cwd, local_triton_model_dir)
    os.makedirs(dir_name, exist_ok=True)

    for obj in objs:
        local_file_name = os.path.join(dir_name, obj.url.replace(s3_root, ""))
        os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
        os.rename(obj.path, local_file_name)
        logging.info(f"Copied {obj.url} to {local_file_name}")
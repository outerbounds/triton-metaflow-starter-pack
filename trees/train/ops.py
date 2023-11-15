import os
from subprocess import check_output
from metaflow import S3, current, Parameter
import logging 
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

SKLEARN_TRITON_CONFIG = """
name: "{deployment_name}"
backend: "fil"
max_batch_size: 8192
input [
 {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {dims} ]
  }}
]
output [
 {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "treelite_checkpoint" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "true" }}
  }},
  {{
    key: "threshold"
    value: {{ string_value: "0.5000" }}
  }}
]
instance_group [{{ kind: KIND_CPU }}]
"""

class ModelStore():

    model_repo = Parameter("model-repo", required=True, help="S3 path to model repo")

    def store_sklearn_estimator(self, model):
        from treelite.sklearn import import_model
        self.serialized_model = import_model(model).serialize_bytes()
        self.deployment_name = f"{current.flow_name}-{current.run_id}"
        self.config = SKLEARN_TRITON_CONFIG.format(
            deployment_name=self.deployment_name,
            dims=model.n_features_in_
        )
        with S3(s3root=self.model_repo) as s3:
            url = s3.put_many([(f"{self.deployment_name}/1/checkpoint.tl", self.serialized_model),
                               (f"{self.deployment_name}/config.pbtxt", self.config)])
            msg = f"The model and its Triton config has deployed at {url}"
            logging.info(msg)



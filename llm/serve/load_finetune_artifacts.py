from metaflow import S3, Flow
import os
from tarfile import TarFile
from io import BytesIO
import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def extract_tar_bytes(tar_bytes, path):
    buf = BytesIO(tar_bytes)
    with TarFile(mode="r", fileobj=buf) as tar:
        tar.extractall(path=path)


def get_triton_repo_from_s3(
    run_id,
    s3_root="s3://outerbounds-datasets/triton/llama2/",
    flow_name="FinetuneLlama",
    triton_model_dir="triton",
):
    with S3(s3root=s3_root) as s3:
        obj = s3.get(f"{flow_name}-{run_id}")
        logging.info(f"Extracting contents of {obj.url} to ./{triton_model_dir}")
        extract_tar_bytes(obj.blob, triton_model_dir)
        logging.info(f"🥳🦙🥳 Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-id", type=str, default="212266")
    args = parser.parse_args()
    get_triton_repo_from_s3(run_id=args.run_id)
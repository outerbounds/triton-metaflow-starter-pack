from metaflow import S3, Flow
import os
import sys
import logging
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def get_triton_repo_from_s3(
    run_id,
    s3_root="s3://outerbounds-datasets/triton/tree-models/",
    flow_name="FraudClassifierTreeSelection",
    local_triton_model_dir="triton",
):

    with S3(s3root=s3_root) as s3:
        objs = s3.get_recursive(f"{flow_name}-{run_id}")

        cwd = os.getcwd()
        dir_name = os.path.join(cwd, local_triton_model_dir)
        os.makedirs(dir_name, exist_ok=True)

        for obj in objs:
            local_file_name = os.path.join(dir_name, obj.url.replace(s3_root, ""))
            os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
            os.rename(obj.path, local_file_name)
            logging.info(f"Copied {obj.url} to {local_file_name}")


def get_test_dataset(flow_name="FraudClassifierTreeSelection", batch_sz=10):
    run = Flow(flow_name).latest_successful_run
    return run.data.X_test_full[:batch_sz], run.data.y_test_full[:batch_sz]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--run-id", type=str, required=True)
    args = parser.parse_args()
    get_triton_repo_from_s3(args.run_id)

from metaflow import S3, Flow
import os

def unpack_triton_artifacts(
    s3_root = "s3://outerbounds-datasets/triton/tree-models/",
    flow_name = "FraudClassifierTreeSelection"
):
    latest_successful_run_id = Flow(flow_name).latest_successful_run.id
    with S3(s3root=s3_root) as s3:
        objs = s3.get_recursive(f"{flow_name}-{latest_successful_run_id}")

        cwd = os.getcwd()
        dir_name = os.path.join(cwd, "triton")
        os.makedirs(dir_name, exist_ok=True)

        for obj in objs:
            url = obj.url
            tmp_path = obj.path
            local_file_name = os.path.join(dir_name, url.replace(s3_root, ""))
            os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
            os.rename(tmp_path, local_file_name)

if __name__ == '__main__':
    unpack_triton_artifacts()
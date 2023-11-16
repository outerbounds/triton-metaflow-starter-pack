from metaflow import S3, Flow
import os


def get_triton_repo_from_s3(
    s3_root="s3://outerbounds-datasets/triton/tree-models/",
    flow_name="FraudClassifierTreeSelection",
    triton_model_dir="triton",
):
    latest_successful_run_id = Flow(flow_name).latest_successful_run.id
    with S3(s3root=s3_root) as s3:
        objs = s3.get_recursive(f"{flow_name}-{latest_successful_run_id}")

        cwd = os.getcwd()
        dir_name = os.path.join(cwd, triton_model_dir)
        os.makedirs(dir_name, exist_ok=True)

        for obj in objs:
            url = obj.url
            tmp_path = obj.path
            local_file_name = os.path.join(dir_name, url.replace(s3_root, ""))
            os.makedirs(os.path.dirname(local_file_name), exist_ok=True)
            os.rename(tmp_path, local_file_name)


def get_test_dataset(flow_name="FraudClassifierTreeSelection", batch_sz=10):
    run = Flow(flow_name).latest_successful_run
    return run.data.X_test_full[:batch_sz], run.data.y_test_full[:batch_sz]


if __name__ == "__main__":
    get_triton_repo_from_s3()

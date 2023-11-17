from sklearn.ensemble import RandomForestClassifier
from metaflow import S3, current
import argparse
import numpy as np
import os

N_SAMPLES = 100000
N_FEATURES = 30

def main(
  store_in_local_repo = True,
  s3_root = "s3://outerbounds-datasets/triton/tree-models-benchmark/" # if not store_in_local_repo
):

    print("Creating benchmark models and storing {}".format("locally" if store_in_local_repo else "in s3"))

    # make model
    rf_clf = RandomForestClassifier(
        random_state=0,
        criterion="gini",
        max_depth=2,
        n_estimators=50,
    )

    # make data
    features, targets = np.random.random((N_SAMPLES, N_FEATURES)), np.choice([0, 1], N_SAMPLES)
    rf_clf.fit(features, targets)

    ###########################################
    ### save with treelite for basic triton ###
    ###########################################

    from treelite.sklearn import import_model
    model = import_model(rf_clf)
    serialized_model = model.serialize_bytes()

    # set params
    deployment_name = "triton"
    config = """
    name: "basic-triton-sklearn"
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
      }}
    ]
    instance_group [{{ kind: KIND_CPU }}]
    """

    if store_in_local_repo:
        model_repo_path = os.path.join(os.getcwd(), "basic-triton", "model-repo")
        model_version_path = os.path.join(model_repo_path, "1")
        if not os.path.exists(model_repo_path):
            os.makedirs(model_repo_path)
        if not os.path.exists(model_version_path):
            os.makedirs(model_version_path)
        with open(os.path.join(model_version_path, "checkpoint.tl"), "wb") as f:
            f.write(serialized_model)
        with open(os.path.join(model_repo_path, "config.pbtxt"), "w") as f:
            f.write(config.format(dims=features.shape[1]))
    else:
        with S3(s3root=s3_root) as s3:
            url = s3.put_many(
                [
                    (f"{deployment_name}/1/checkpoint.tl", serialized_model),
                    (f"{deployment_name}/config.pbtxt", config.format(dims=features.shape[1])),
                ]
            )

    ##########################################
    ### save with pickle for basic fastapi ###
    ##########################################

    import pickle
    model_bytes = pickle.dumps(rf_clf)

    deployment_name = "fastapi"

    if store_in_local_repo:
        model_repo_path = os.path.join(os.getcwd(), "basic-fastapi", "model-repo")
        if not os.path.exists(model_repo_path):
            os.makedirs(model_repo_path)
        with open(os.path.join(model_repo_path, "model.pkl"), "wb") as f:
            f.write(model_bytes)
    else:
        with S3(s3root=s3_root) as s3:
            url = s3.put(f"{deployment_name}/model.pkl", model_bytes)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--local", action="store_true", required=False)
    parser.add_argument("-s", "--s3_root", action="store_true", required=False)
    args = parser.parse_args()

    main(store_in_local_repo=args.local, s3_root=args.s3_root)
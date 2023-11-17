# Run the training/tuning workflows manually

## Environment setup
```
$ mamba env create -f env_unpinned.yml
$ mamba activate triton-ob-dev
$ cd trees
```

## Deploy model to cloud storage
```
$ export S3_URI=s3://outerbounds-datasets/triton/tree-models/
$ python train/flow.py --environment=pypi run --model-repo $S3_URI
```

# Set up the server

## Get the models

### Path 1: Manually unpack training artifacts

#### Install AWS CLI
[Follow instructions based on your server OS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

#### Create model repository manually
```
$ export MODEL_REPO=triton
$ mkdir $MODEL_REPO
```

#### Move artifacts from S3 to repository
The following script will download the Triton model repository, including the config and [treelite](https://treelite.readthedocs.io/en/latest/) artifact from S3.
```
$ python load_train_artifacts.py
```

## Launch Triton Server from NGC Container
The following runs the NVIDIA Triton Server in a Docker container running in "detached" mode. Run this command:
```
$ export MODEL_REPO=triton
$ docker run --rm -d -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/$MODEL_REPO:/$MODEL_REPO --name tritonserver nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/$MODEL_REPO
```

After a few seconds, you can check the container is running:
```
$ docker ps
```
and then check the server started correctly:
```
$ docker logs tritonserver

...
I1116 12:19:52.021379 1 grpc_server.cc:2513] Started GRPCInferenceService at 0.0.0.0:8001
I1116 12:19:52.022197 1 http_server.cc:4497] Started HTTPService at 0.0.0.0:8000
I1116 12:19:52.064891 1 http_server.cc:270] Started Metrics Service at 0.0.0.0:8002
```

## Make requests
Once the container is set up, you can simulate a client making an API request.
First, install the Triton Python client:
```
$ pip install tritonclient[all]
```
Then, you can copy the `serve/client.py` script to the server and simulate another application requests predictions from the API:
```
$ export FLOW_NAME=FraudClassifierTreeSelection
$ export RUN_ID=... 
$ python client.py -m $FLOW_NAME-$RUN_ID
```
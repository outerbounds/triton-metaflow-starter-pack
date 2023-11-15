# Run the training/tuning workflows manually

## Environment setup
```
mamba env create -f env.yml
mamba activate triton-ob-dev
```

## Deploy model to cloud storage
```
export S3_URI=s3://outerbounds-datasets/triton/tree-models/
python flow.py --environment=pypi run --model-repo $S3_URI
```

# Set up the server

## 1. Get the models

### Path 1: Manually unpack training artifacts

#### Install AWS CLI
[Follow instructions based on your server OS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

#### Create model repository manually
```
export MODEL_REPO=triton
mkdir $MODEL_REPO
```

#### Manually move artifacts from S3 to repository
```
aws s3 cp --recursive s3://outerbounds-datasets/triton/tree-models ./$MODEL_REPO
```

### Path 2: Automatically unpack training artifacts
*This approach requires a consistent Metaflow config file across training and server VMs*

The following script will download the triton model repository, including the config and treelite artifact from S3.
```
python inference-server/main.py
```

## Launch Triton Server from NGC Container
```
export MODEL_REPO=triton
docker run --rm --net=host -v ${PWD}/$MODEL_REPO:/$MODEL_REPO nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/$MODEL_REPO
```

## Make requests

To simulate a client, open up the server in a new terminal and use this getting started script.

# Start client
```
docker run -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.10-py3-sdk bash
```
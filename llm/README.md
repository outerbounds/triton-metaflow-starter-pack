# Run the training/tuning workflows manually

## Environment setup
```
$ mamba env create -f env.yml
$ mamba activate triton-ob-dev
```

## Deploy model to cloud storage
```
$ export S3_URI=s3://outerbounds-datasets/triton/llama2/
$ python flow.py --environment=pypi run --model-repo $S3_URI
```

# Set up the server

## Get the model repository

#### Install AWS CLI
[Follow instructions based on your server OS](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

#### Create model repository manually
```
$ export MODEL_REPO=triton
$ mkdir $MODEL_REPO
```

#### Manually move artifacts from S3 to repository

First we download the compressed model repository from S3:
```
$ aws s3 cp --recursive s3://outerbounds-datasets/triton/llama2/ .
```
Then we unpack it - you will need to copy the script in `serve/load_finetune_artifacts.py` into the machine that will run the Triton server.
```
$ python load_finetune_artifacts.py
```

Now look inside the `triton` directory, where you should see one or more model repositories structure like this:
```
| <FLOW_NAME>-<RUN_ID>
----| 1
--------| backend.py
--------| <save_pretrained_path>
------------| model
----------------| adapter_config.json
----------------| adapter_model.bin
------------| tokenizer
----------------| special_tokens_map.json
----------------| tokenizer_config.json
----------------| tokenizer.json
----| config.pbtxt
```

## Launch Triton Server from NGC Container
```
$ export MODEL_REPO=triton
$ docker run --rm --net=host -v ${PWD}/$MODEL_REPO:/$MODEL_REPO nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/$MODEL_REPO

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

To simulate a client, open up the server in a new terminal and use this getting started script.

# Start client

Once the container is set up, you can simulate a client making an API request.

First, install the Triton Python client:
```
$ pip install tritonclient[all]
```

Then, you can copy the `serve/client.py` script to the server and simulate another application requests predictions from the API:
```
from client import *
_ = batch_inference([
    ["I want to use AI technology to help humans and other animals live more fulfilling lives. How can AI help?"],
    ["Write a concise roadmap for how AI can generate abudance without runaway wealth inequality."],
    ["Write a set of fun activities I can do with my nieces."]
])
```
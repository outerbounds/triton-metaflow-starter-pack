# Benchmarking Triton for serving Tree models

This subdirectory contains codes for a benchmark of various ways to serve tree models. 

The benchmark is structured as follows:
- A Python script sends `N=1,000,000` API requests
- Each request contains a NumPy array with 30 numerical features
- The time to respond to the request is measured in the client script, as the time right before the request is sent to the time right after the result is received in the client script.

You can find the included solutions in this table:

| Name | Frontend | Backend | Model Seriazation | Time per query |
| :---: | :---: | :---: | :---: | :---: |
| Basic Triton + RAPIDS | Triton | FIL | Treelite | ？|
| Basic FastAPI | FastAPI | Uvicorn | Pickle | ？|

# Instructions

## Run `create_model.py`

You can either run it directly on the server, and store artifacts locally:
```
python create_model.py -l
```
or you can run it from anywhere with access to push to an S3 bucket:
```
python create_model.py --s3_root s3://outerbounds-datasets/triton/tree-models-benchmark/
```

## Setup the server instance
Clone this repository:
```
git clone https://github.com/outerbounds/triton-metaflow-starter-pack.git
cd triton-metaflow-starter-pack/trees/serving-benchmark
```

Set up the dependencies:
```
pip install -r requirements.txt
```

## Run the benchmark

### Run the Triton Benchmark
```
export MODEL_REPO=basic-triton
docker run --rm -d -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/$MODEL_REPO:/$MODEL_REPO --name tritonserver nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/$MODEL_REPO
python triton_latency_test.py -n 10000
docker stop tritonserver
```

### Run the FastAPI Benchmark

##### Start the server
```
cd basic-fastapi
uvicorn server:app
```

##### Run the test
Open a new terminal while uvicorn is running:
```
python fastapi_latency_test.py
```
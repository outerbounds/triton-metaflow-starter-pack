# python basic-triton/bench.py --results-dir .
# python basic-fastapi/bench.py --results-dir .

##########
# common #
##########
python3 create_model.py

# create sklearn model and serialize it to a file for each row in benchmark table

##########
# triton #
##########
# start triton server

# run benchmark
cd basic-triton
export MODEL_REPO=basic-triton
docker run --rm -d -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/$MODEL_REPO:/$MODEL_REPO --name tritonserver nvcr.io/nvidia/tritonserver:23.10-py3 tritonserver --model-repository=/$MODEL_REPO

docker stop tritonserver

###########
# fastapi #
###########
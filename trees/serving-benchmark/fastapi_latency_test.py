import argparse
import logging
import sys
import time 
import numpy as np
from create_model import N_SAMPLES, N_FEATURES
import json
import requests
from tqdm import tqdm

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def fastapi_predict(arr):
    endpoint_uri_base = "http://127.0.0.1:8000"
    endpoint = "predict?data={}".format(json.dumps(arr.tolist()))
    url = "{}/{}".format(endpoint_uri_base, endpoint)
    response = requests.get(url, verify=False, proxies={'https': endpoint_uri_base})
    return response.json()["prediction"]

def fast_api_main(N=1_000_000):
    data_point = np.random.rand(1, N_FEATURES).astype(np.float32)

    ############################
    ### FastAPI latency test ###
    ############################

    # Warmup    
    for _ in range(10):
        res = fastapi_predict(data_point)

    fastapi_latencies = []
    for _ in tqdm(range(N)):
        t0 = time.time()
        res = fastapi_predict(data_point)
        t1 = time.time()
        fastapi_latencies.append(t1-t0)
    
    return fastapi_latencies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--results-dir', type=str, default='results'
    )
    parser.add_argument(
        '-n', '--num-samples', type=int, default=1_000_000
    )
    args = parser.parse_args()
    fastapi_latencies = fast_api_main(args.num_samples)
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    np.save(f'{args.results_dir}/fastapi_latencies.npy', fastapi_latencies)
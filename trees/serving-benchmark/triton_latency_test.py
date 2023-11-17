import argparse
import logging
import sys
import time 
import numpy as np
from create_model import N_SAMPLES, N_FEATURES
import tritonclient.http as triton_http
from tqdm import tqdm
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

def triton_predict(client, model_name, arr, protocol="http"):
    batch_sz, n_features = arr.shape
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        logging.info(
            f"⚠️ Input array to `triton_predict` was not of type np.float32. Casting to np.float32."
        )

    if protocol == "http":
        triton_input = triton_http.InferInput(
            "input__0", (batch_sz, n_features), "FP32"
        )
        triton_output = triton_http.InferRequestedOutput("output__0")
    elif protocol == "grpc":
        triton_input = triton_grpc.InferInput(
            "input__0", (batch_sz, n_features), "FP32"
        )
        triton_output = triton_grpc.InferRequestedOutput("output__0")

    triton_input.set_data_from_numpy(arr)
    response = client.infer(
        model_name, model_version="1", inputs=[triton_input], outputs=[triton_output]
    )
    return response.as_numpy("output__0")

def triton_main(N=1_000_000):
    data_point = np.random.rand(1, N_FEATURES).astype(np.float32)

    ###########################
    ### Triton latency test ###
    ###########################
    url = "localhost:8000" # http
    model_name = "model-repo"

    client = triton_http.InferenceServerClient(url, verbose=False, concurrency=1)
    assert client.is_server_ready(), "Triton server is not ready!"
    assert client.is_model_ready(model_name), f"Triton model {model_name} is not ready!"

    # Warmup    
    for _ in range(10):
        res = triton_predict(client, model_name, data_point)

    triton_latencies = []
    for _ in tqdm(range(N)):
        t0 = time.time()
        res = triton_predict(client, model_name, data_point)
        t1 = time.time()
        triton_latencies.append(t1-t0)
    
    return triton_latencies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--results-dir', type=str, default='results'
    )
    parser.add_argument(
        '-n', '--num-samples', type=int, default=1_000_000
    )
    args = parser.parse_args()
    triton_latencies = triton_main(args.num_samples)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    np.save(f'{args.results_dir}/triton_latencies.npy', triton_latencies)
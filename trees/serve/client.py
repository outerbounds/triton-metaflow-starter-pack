from tritonclient.utils import *
import tritonclient.http as triton_http
import tritonclient.grpc as triton_grpc
from tritonclient import utils as triton_utils

import argparse
import logging
import time
import sys
import os

# from load_train_artifacts import get_test_dataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def is_triton_ready(model_name):
    server_start = time.time()
    while True:
        try:
            if client.is_server_ready() and client.is_model_ready(model_name):
                return True
        except triton_utils.InferenceServerException:
            pass
        if time.time() - server_start > TIMEOUT:
            print(
                "Server was not ready before given timeout. Check the logs below for possible issues."
            )
            os.system("docker logs tritonserver")
            return False
        time.sleep(1)


def triton_predict(client, model_name, arr, protocol="http"):
    batch_sz, n_features = arr.shape
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
        logging.info(
            f"⚠️  Input array to `triton_predict` was not of type np.float32. Casting to np.float32."
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


if __name__ == "__main__":
    # Client inputs
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-host",
        "--host",
        type=str,
        required=False,
        help="Inference server host. Default is localhost.",
        default="localhost",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        required=False,
        help="Inference server port. Default is 8000 for http.",
        default="8000",
    )
    parser.add_argument(
        "-m", "--model-name", required=True, help="Enable verbose output"
    )
    parser.add_argument(
        "-mv",
        "--model-version",
        required=False,
        help="Specify the model version to use from the triton repository",
        default=1,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        required=False,
        help="Concurrency for the client. Default is 2.",
        default=2,
    )
    args = parser.parse_args()

    # Initialize triton client
    url = f"{args.host}:{args.port}"
    client = triton_http.InferenceServerClient(url, args.verbose, args.concurrency)

    # Server startup health check
    assert (
        client.is_server_ready()
    ), "❌ Triton server is not ready! Exiting Python program."
    logging.info("✅ Triton server is ready! ")
    logging.info(f"Checking if model {args.model_name} is ready...")
    assert client.is_model_ready(
        args.model_name
    ), f"❌ Model {args.model_name} is not ready! Exiting client.py program."
    logging.info(f"✅ Model {args.model_name} is ready!")

    # basic API request
    batch_sz = 2

    # get_test_dataset assumes you can pull Metaflow results from the server.
    # here we manually construct data from scratch to show form of the dataset.
    # X_test, y_test = get_test_dataset(batch_sz = batch_sz)
    X_test = np.array(
        [
            [
                4.15050000e04,
                -1.65265066e01,
                8.58497180e00,
                -1.86498532e01,
                9.50559352e00,
                -1.37938185e01,
                -2.83240430e00,
                -1.67016943e01,
                7.51734390e00,
                -8.50705864e00,
                -1.41101844e01,
                5.29923635e00,
                -1.08340065e01,
                1.67112025e00,
                -9.37385858e00,
                3.60805642e-01,
                -9.89924654e00,
                -1.92362924e01,
                -8.39855199e00,
                3.10173537e00,
                -1.51492344e00,
                1.19073869e00,
                -1.12767001e00,
                -2.35857877e00,
                6.73461329e-01,
                -1.41369967e00,
                -4.62762361e-01,
                -2.01857525e00,
                -1.04280417e00,
                3.64190000e02,
            ],
            [
                4.42610000e04,
                3.39812064e-01,
                -2.74374524e00,
                -1.34069511e-01,
                -1.38572931e00,
                -1.45141332e00,
                1.01588659e00,
                -5.24379057e-01,
                2.24060376e-01,
                8.99746005e-01,
                -5.65011684e-01,
                -8.76702573e-02,
                9.79426988e-01,
                7.68828168e-02,
                -2.17883812e-01,
                -1.36829588e-01,
                -2.14289209e00,
                1.26956065e-01,
                1.75266151e00,
                4.32546224e-01,
                5.06043885e-01,
                -2.13435844e-01,
                -9.42525025e-01,
                -5.26819175e-01,
                -1.15699190e00,
                3.11210510e-01,
                -7.46646679e-01,
                4.09958027e-02,
                1.02037825e-01,
                5.20120000e02,
            ],
        ]
    )
    y_test = np.array([1, 0])
    preds = triton_predict(client, args.model_name, X_test)
    batch_accuracy = 100 * np.sum(preds == y_test) / batch_sz
    logging.info(f"Accuracy on {batch_sz} predictions: {batch_accuracy:.2f}%")

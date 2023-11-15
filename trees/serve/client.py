from tritonclient.utils import *
import tritonclient.http as httpclient
import argparse
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# with httpclient.InferenceServerClient(
#     url="localhost:8000", verbose=False, concurrency=8
# ) as client:

# s3_root = "s3://outerbounds-datasets/triton/tree-models/"
# flow_name = "FraudClassifierTreeSelection"
# latest_successful_run_id = Flow(flow_name).latest_successful_run.id


if __name__ == '__main__':

    # client inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-h', '--host', type=str, required=False, help='Inference server host. Default is localhost.', default="localhost")
    parser.add_argument('-p', '--port', type=str, required=False, help='Inference server port. Default is 8000.', default="8000")  
    parser.add_argument('-m', '--model-name', required=True, help='Enable verbose output')
    parser.add_argument('-mv', '--model-version', required=False, help='Enable verbose output', default=1)
    parser.add_argument('-v', '--verbose', action="store_true", required=False, help='Enable verbose output')
    parser.add_argument('-c', '--concurrency', type=int, required=False, help='Concurrency for the client. Default is 8.', default=8)
    args = parser.parse_args()

    # initialize triton client
    url = f"{args.host}:{args.port}"
    client = httpclient.InferenceServerClient(url, args.verbose, args.concurrency)

    # startup checks
    assert client.is_server_ready(), "❌ Triton server is not ready! Exiting Python program."
    logging.info("✅ Triton server is ready! ")
    logging.info(f"Checking if model {args.model_name} is ready...")
    assert client.is_model_ready(args.model_name), f"❌ Model {args.model_name} is not ready! Exiting Python program."
    logging.info(f"✅ Model {args.model_name} is ready!")

    # API request
    # response = client.infer(
    #     args.model_name,
    #     model_version=args.model_version, 
    #     inputs=inputs, 
    #     outputs=outputs
    # )
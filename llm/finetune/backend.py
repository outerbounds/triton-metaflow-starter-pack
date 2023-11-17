from setup_env import main as _setup_env

_setup_env()
# Above two lines create a virtual environment and install the dependencies.
# This should be cached somewhere. This makes it happen on each server start.

import app
import os
import json
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import huggingface_hub
from threading import Thread
from tarfile import TarFile
from io import BytesIO
from metaflow import S3

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

flow_artifacts_dir = sorted(os.listdir("/triton"))[-1]
flow_name = flow_artifacts_dir.split("-")[0]
run_id = flow_artifacts_dir.split("-")[1]

local_path_prefix = f"/triton/{flow_name}-{run_id}"
save_pretrained_path = "llama-2-7b-dolly15k"
checkpoint_model_path = "%s/1/%s/model" % (local_path_prefix, save_pretrained_path)
checkpoint_tokenizer_path = "%s/1/%s/tokenizer" % (local_path_prefix, save_pretrained_path)


def extract_tar_bytes(tar_bytes, path):
    buf = BytesIO(tar_bytes)
    with TarFile(mode="r", fileobj=buf) as tar:
        tar.extractall(path=path)


def format_prompt(example):
    return f"""### INSTRUCTION: {example['instruction']}

    ### CONTEXT: {example['context']}
                            
    ### RESPONSE: {example['response']}
    """


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint_model_path).to(
            "cuda"
        )
        self.task = "text-generation"
        self.max_length = 200

    def get_prompt(self, user_input: str, context: str):
        return format_prompt(
            {
                "instruction": user_input,
                "context": context,
                "response": "",
                # start the response the LLM should continue with the user input
                # these are the supervised learning labels in the dolly15k instruction tuning format
            }
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Decode the Byte Tensor into Text
            inputs = pb_utils.get_input_tensor_by_name(request, "prompt")
            inputs = inputs.as_numpy()

            context = []  # TODO: retrieve this from a RAG pipeline!

            prompts = [self.get_prompt(i[0].decode(), context) for i in inputs]
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                "cuda"
            )

            output_sequences = self.model.generate(
                **inputs,
                do_sample=True,
                max_length=self.max_length,
                temperature=0.01,
                top_p=1,
                top_k=20,
                repetition_penalty=1.1,
            )

            output = self.tokenizer.batch_decode(
                output_sequences, skip_special_tokens=True
            )

            # Encode text as byte tensor to send in response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "generated_text",
                        np.array([[o.encode() for o in output]]),
                    )
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self, args):
        self.generator = None

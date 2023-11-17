from metaflow import FlowSpec, step, pypi, batch, environment, S3, current, Parameter
from ops import ModelStore

N_GPU = 1
N_CPU = 8 * N_GPU
MEMORY = 16000 * N_GPU
IMAGE = "eddieob/llama2-finetune-qlora:latest"


class FinetuneLlama(FlowSpec, ModelStore):

    dataset_fraction = Parameter("-d", "dataset-fraction", default=0.05)

    @step
    def start(self):
        self.next(self.train)

    @environment(
        vars={
            "CUDA_VISIBLE_DEVICES": ",".join([str(i) for i in list(range(N_GPU))]),
            "NCCL_SOCKET_IFNAME": "eth0",
        }
    )
    @batch(gpu=N_GPU, cpu=N_CPU, memory=MEMORY, image=IMAGE)
    @step
    def train(self):
        from params import model_path
        from model import main

        main(dataset_fraction=self.dataset_fraction)
        self.store_transformer(model_path)
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    FinetuneLlama()

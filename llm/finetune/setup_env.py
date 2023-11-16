import subprocess

requirements = [
    "app",
    "torch",
    "scipy",
    "metaflow",
    "accelerate",
    "bitsandbytes",
    "peft",
    "trl",
    "transformers",
    "huggingface_hub",
]


def main():
    for requirement in requirements:
        subprocess.run(["pip", "install", requirement])


if __name__ == "__main__":
    main()

import argparse
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ...
    args = parser.parse_args()
    get_triton_repo_from_s3(args.run_id)
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

# data viz config
GREEN = '#37795D'
PURPLE = '#5460C0'
BACKGROUND = '#F4EBE6'
colors = [GREEN, PURPLE]
custom_params = {
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.facecolor':BACKGROUND, 'figure.facecolor': BACKGROUND, 
    'figure.figsize':(8, 8)
}
sns_palette = sns.color_palette(colors, len(colors))
sns.set_theme(style='ticks', rc=custom_params)

trial_name_map = {
    "fastapi": "FastAPI + Uvicorn",
    "triton": "Triton + FIL",
}

def get_stats(result_dir):
    df = pd.DataFrame()
    for res in os.listdir(result_dir):
        if not res.endswith('.npy'):
            continue
        trial_name = res.split('_')[0]
        latency_data = np.load(f'{result_dir}/{res}')
        df_trial = pd.DataFrame({
            'trial_name': [trial_name]*len(latency_data),
            'latency': latency_data
        })
        df = pd.concat([df, df_trial])
    return df

def make_plot(stats, outdir, num_samples):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.boxplot(data=stats, x="trial_name", y="latency", ax=ax, palette=sns_palette)
    ax.set_yscale('log')
    ax.set_ylabel("Latency (seconds)", fontsize=16)
    ax.set_xlabel("Serving System", fontsize=16)
    ax.set_xticklabels([trial_name_map[tick.get_text()] for tick in ax.get_xticklabels()], fontsize=12)
    ax.set_title(f"Average Sklearn request latency over {num_samples} samples", fontsize=20)
    plt.savefig(outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--result-dir", type=str, default="results")
    parser.add_argument("-o", "--output-dir", type=str, default="latency-bar-chart.png")
    parser.add_argument("-n", "--num-samples", type=int, required=True)
    args = parser.parse_args()

    stats = get_stats(args.result_dir)
    make_plot(stats, args.output_dir, args.num_samples)
import pandas as pd
import numpy as np

def process_csv(codec):
    # read csv
    df = pd.read_csv(f'outputs/csv_outputs/eval_{codec}.csv')

    # remove extra columns
    df = df[[col for col in df.columns if '_MIN' not in col and '_MAX' not in col]]
    df.columns = [col.split('/')[-1] for col in df.columns]

    # extract data from csv
    min = 300
    final_df = pd.DataFrame(np.zeros((300, len(df.columns[1:]))), columns=df.columns[1:].to_numpy())
    for col in df.columns[1:]:
        col_data = df[col].dropna().to_numpy()
        if len(col_data) < min:
            min = len(col_data)
            print(f"{codec}: {col} has {len(col_data)} frames")
        final_df[col] = np.pad(col_data, (0, 300 - len(col_data)), constant_values=col_data[-1])
    arr = final_df.to_numpy()

    # get max reward for each rollout
    max_reward = arr.max(axis=0)

    # optional max reward plot
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 5))
    # plt.hist(max_reward, bins=500, alpha=0.7, color='blue', edgecolor='black')
    # plt.title(f'Histogram of max_reward for {codec}')
    # plt.xlabel('Max Value')
    # plt.ylabel('Frequency')
    # plt.xlim(0, 1)
    # plt.ylim(0, 100)
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig(f'outputs/eval/new_compression_100/csv_outputs/hist_{codec}.png')

    print(f'{codec} success rate: {np.count_nonzero(max_reward >= 0.95)} or {np.count_nonzero(arr[-1, :] >= 0.95)} / {arr.shape[1]}, max reward: {np.mean(max_reward):.4f} ± {np.std(max_reward):.4f}')

for codec in ['auto', 'rawvideo', 'libaom-av1', 'libx264', 'libx265', 'ffv1']:
    process_csv(codec)

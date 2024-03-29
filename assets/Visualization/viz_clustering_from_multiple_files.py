import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

directory = './CURE-parallelization-main/assets/Visualization'

file_paths = [
    f'{directory}/cluster_result_500data_procs1.csv',
    f'{directory}/cluster_result_500data_procs2.csv',
    f'{directory}/cluster_result_500data_procs4.csv',
    f'{directory}/cluster_result_500data_procs8.csv'
]

dataframes = [pd.read_csv(file) for file in file_paths]

fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle('Cluster Results for 500 Data Points Across Different Processes')

color_map = {
    0: 'blue',
    -1: 'green',
    1: (1.0, 0.0, 0.0, 0.5),
    2: (1.0, 1.0, 0.0, 0.5)
}

for ax, df, procs in zip(axs.flatten(), dataframes, [1, 2, 4, 8]):
    unique_labels = df.iloc[:, 2].unique()
    for label in unique_labels:
        subset = df[df.iloc[:, 2] == label]
        ax.scatter(subset.iloc[:, 0], subset.iloc[:, 1], color=color_map[label], label=f'Cluster {label}')
    
    ax.set_title(f'Procs: {procs}')
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: int(t[0].split(' ')[-1])))
    ax.legend(handles, labels, title='Cluster')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('cluster_results_without.png', dpi=300)
plt.show()

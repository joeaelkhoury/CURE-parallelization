import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load your data into Pandas DataFrames
# For demonstration, replace these with the paths to your actual files

directory = './CURE-parallelization-main/assets/Visualization'
file_paths = [
    f'{directory}/cluster_result_2000data_procs1.csv',
    f'{directory}/cluster_result_2000data_procs2.csv',
    f'{directory}/cluster_result_2000data_procs4.csv',
    f'{directory}/cluster_result_2000data_procs8.csv'
]


dataframes = [pd.read_csv(file) for file in file_paths]

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
fig.suptitle('Cluster Results for 2000 Data Points Across Different Processes')

# Define a colormap
cmap = plt.cm.get_cmap('viridis')

for ax, df, procs in zip(axs.flatten(), dataframes, [1, 2, 4, 8]):
    # Determine unique labels and assign colors
    unique_labels = df.iloc[:, 2].unique()
    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        # Filter rows by label and plot
        subset = df[df.iloc[:, 2] == label]
        ax.scatter(subset.iloc[:, 0], subset.iloc[:, 1], color=color, label=f'Cluster {int(label)}')
    
    ax.set_title(f'Procs: {procs}')
    ax.legend(title='Cluster')
    # ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('cluster_results.png', dpi=300)
plt.show()


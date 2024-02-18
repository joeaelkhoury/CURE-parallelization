import os
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Directory containing the files

directory = './CURE-parallelization-main/benchmark'

# Regex patterns to find files and extract data
file_pattern = re.compile(r'output_(\d+)(k|M)?data_procs(\d+)\.txt')
# Updated regex pattern to match your file format
time_pattern = re.compile(r'Clustering took ([\d.]+) seconds')

# Store execution times in a dict of dicts format {data_points: {procs: time}}
execution_times = {}
display_format = {}

def convert_and_store_data_points(data_points, multiplier):
    if multiplier == 'k':
        numeric_value = int(data_points) * 1000
    elif multiplier == 'M':
        numeric_value = int(data_points) * 1000000
    else:
        numeric_value = int(data_points)
    display_format[numeric_value] = data_points + (multiplier if multiplier else '')
    return numeric_value

def calculate_metrics_and_print_time(exec_times):
    for data_points in sorted(exec_times.keys()):
        procs_times = exec_times[data_points]
        base_time = procs_times[min(procs_times.keys())]
        formatted_data_points = display_format[data_points]
        print(f"Data Points: {formatted_data_points}")
        for procs in sorted(procs_times.keys()):
            time = procs_times[procs]
            speedup = base_time / time
            efficiency = speedup / procs
            print(f"Procs: {procs}, Execution Time: {time:.2f} seconds, Speedup: {speedup:.2f}, Efficiency: {efficiency:.2f}")

def plot_metrics(exec_times):
    plt.figure(figsize=(12, 6))
    
    # Number of datasets
    num_datasets = len(exec_times)
    
    # Generate a color map
    colors = cm.rainbow(np.linspace(0, 1, num_datasets))
    
    # Prepare plotting data
    plotting_data_speedup = []
    plotting_data_efficiency = []
    
    for data_points, procs_times in exec_times.items():
        procs = sorted(procs_times.keys())
        base_time = procs_times[min(procs)]
        speedups = [base_time / procs_times[p] for p in procs]
        efficiencies = [(base_time / procs_times[p]) / p for p in procs]
        label = f'{display_format[data_points]} data points'
        plotting_data_speedup.append((data_points, procs, speedups, label))
        plotting_data_efficiency.append((data_points, procs, efficiencies, label))
    
    # Sort by data_points for plotting
    plotting_data_speedup.sort(key=lambda x: x[0])
    plotting_data_efficiency.sort(key=lambda x: x[0])

    # Plot for Speedup
    plt.subplot(1, 2, 1)
    for index, (_, procs, speedups, label) in enumerate(plotting_data_speedup):
        plt.plot(procs, speedups, marker='o', label=label, color=colors[index])
    plt.xlabel('Number of Processes')
    plt.ylabel('Speedup')
    plt.title('Speedup vs. Number of Processes')
    plt.legend()

    # Plot for Efficiency
    plt.subplot(1, 2, 2)
    for index, (_, procs, efficiencies, label) in enumerate(plotting_data_efficiency):
        plt.plot(procs, efficiencies, marker='o', label=label, color=colors[index])
    plt.xlabel('Number of Processes')
    plt.ylabel('Efficiency')
    plt.title('Efficiency vs. Number of Processes')
    plt.legend()

    plt.tight_layout()
    plt.savefig('benchmark.png',dpi = 300)
    plt.show()

def main():
    # Ensure the directory variable points to the correct path
    for filename in os.listdir(directory):
        match = file_pattern.match(filename)
        if match:
            data_points, multiplier, procs = match.groups()
            numeric_data_points = convert_and_store_data_points(data_points, multiplier)
            procs = int(procs)

            with open(os.path.join(directory, filename), 'r') as file:
                for line in file:
                    time_match = time_pattern.search(line)
                    if time_match:
                        execution_time = float(time_match.group(1))
                        if numeric_data_points not in execution_times:
                            execution_times[numeric_data_points] = {}
                        execution_times[numeric_data_points][procs] = execution_time
                        break  # Assumes first match is the desired one

    calculate_metrics_and_print_time(execution_times)
    plot_metrics(execution_times)

if __name__ == "__main__":
    main()
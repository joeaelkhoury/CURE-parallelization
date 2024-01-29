
# CURE Clustering Algorithm (MPI)

This program implements the CURE (Clustering Using Representatives) clustering algorithm, utilizing MPI (Message Passing Interface) for parallel processing. The CURE algorithm is particularly effective for large datasets and is designed to handle outliers well.

## Features

- **CURE Clustering**: Implements the CURE algorithm, which identifies clusters based on a representative points approach.
- **Distance Measures**: Supports Euclidean and Manhattan distance calculations.
- **Parallel Processing**: Leverages MPI to distribute the clustering process across multiple nodes.
- **Dynamic Thresholds**: Customizable thresholds for outlier detection and cluster splitting.

## Prerequisites

- An MPI library (MPICH)
- A C compiler
- Access to a compute environment with MPI support

## Compilation

Compile the code using `mpicc`:

```bash
mpicc -g -Wall -fopenmp CURE.c -o cure_cluster -lm
```

The `-fopenmp` flag is included for OpenMP support if required.

## Running the Program

Execute the program using `mpiexec`:

```bash
mpiexec -n <number_of_processes> ./cure_cluster <input_file>
```

- Replace `<number_of_processes>` with the desired number of MPI processes.
- `<input_file>` should be the path to your data file.

## Input Data

The input should be a CSV file where each line represents a data point in the format:

```
x_coordinate, y_coordinate
```

## Output

The program outputs the clustering result, which includes cluster assignments for each data point. The output is printed to the console and can be redirected to a file if needed.

## Configuration

Modify the following definitions in the code to change the clustering behavior:

```c
#define SOME_DEFINED_THRESHOLD 0.5
#define ANOTHER_DEFINED_THRESHOLD 0.3
```

- `SOME_DEFINED_THRESHOLD` is used for outlier detection.
- `ANOTHER_DEFINED_THRESHOLD` is used for deciding when to split clusters.

## Key Functions

- `compute_distance`: Calculates Euclidean distance.
- `manhattan_distance`: Calculates Manhattan distance.
- `hierarchical_clustering`: Core function for applying the CURE algorithm.

## Notes

- Ensure that the MPI environment is correctly configured on your system.
- The program's performance and efficiency depend on the configuration of the MPI environment and the characteristics of the input data.

## Author

**Joe El Khoury**  
**Yusuke Sugihara**


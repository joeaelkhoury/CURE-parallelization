# CURE Clustering Implementation Using MPI

This document provides a detailed explanation of the CURE (Clustering Using Representatives) clustering implementation using the Message Passing Interface (MPI) for parallel processing. The implementation involves several critical steps: initialization, data preparation and distribution, local and global clustering operations, label assignment, and result saving. The code is designed to run efficiently on a distributed computing environment, leveraging MPI for inter-process communication.

## Diagrams Overview

Throughout this document, we'll reference several diagrams that illustrate various parts of the implementation. These diagrams are located in the GitHub repository under `/main/assets/Diagrams`.

- **Process**: ![MPI Initialization Diagram](https://github.com/joeaelkhoury/CURE-parallelization/blob/main/assets/Diagrams/full.png)

- **MPI Communication**: ![MPI Initialization Diagram](https://github.com/joeaelkhoury/CURE-parallelization/blob/main/assets/Diagrams/all_process.png)


## Overview

The implementation is structured as follows:

1. **MPI Initialization**: Set up the MPI environment for parallel processing.
2. **Data Preparation**: Read the dataset and optionally stratify shuffle it for balanced distribution.
3. **Data Distribution**: Distribute data points among MPI processes.
4. **Local Clustering**: Perform clustering on local data segments.
5. **Global Clustering**: Merge local clusters into global clusters on the root process.
6. **Label Assignment**: Assign labels to data points based on the clustering results.
7. **Result Saving**: Save the clustering results to a file.
8. **MPI Finalization**: Clean up and exit the MPI environment.

### MPI Initialization

At the beginning of the main function, MPI is initialized using `MPI_Init`. This step prepares the program for parallel execution across multiple processes.

```c
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
MPI_Comm_size(MPI_COMM_WORLD, &world_size);
```

### Data Preparation and Broadcasting

The root process reads the input dataset from a file and, if needed, performs a stratified shuffle to ensure a balanced distribution of data points. The size of the dataset is then broadcasted to all processes to prepare them for data reception.

```c
if (world_rank == 0) {
    data = read_data("f.txt", &num_points, world_rank);
    stratified_shuffle_data(data, num_points, num_strata_x, num_strata_y);
}
MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

### Data Distribution

Using `MPI_Scatterv`, the dataset is divided among the processes. Each process receives a segment of the data to work on locally.

```c
MPI_Scatterv(data, sendcounts, displs, MPI_POINT, local_data, sendcounts[world_rank], MPI_POINT, 0, MPI_COMM_WORLD);
```

### Local Clustering

Each process performs clustering on its segment of the data. This involves initializing clusters, performing iterative merging of clusters based on proximity, and adjusting representatives towards the centroid.

### Global Clustering

The root process gathers the local clusters from all processes and performs a final round of merging to reduce the number of clusters to the desired count.

### Label Assignment

Each process assigns labels to its local data points based on the clustering results. This step may involve determining the nearest cluster representative for each point.

### Result Saving

The root process gathers labels from all processes and saves the final clustering results to a file. The file includes coordinates of data points along with their assigned cluster labels.

```c
if (world_rank == 0) {
    FILE* file = fopen(output_filename, "w");
    for (int i = 0; i < num_points; i++) {
        fprintf(file, "%f,%f,%d\n", data[i].x, data[i].y, global_labels[i]);
    }
    fclose(file);
}
```

### MPI Finalization

Finally, the MPI environment is cleaned up using `MPI_Finalize`, and the program exits. This step ensures that all MPI-related resources are properly released.

```c
MPI_Type_free(&MPI_POINT);
MPI_Finalize();
```

## Compilation and Execution

The code is compiled using `mpicc` and executed across multiple processes using `mpiexec`. The compilation and execution process is managed by a PBS (Portable Batch System) script, which specifies the required resources and execution parameters.

```bash
mpicc -g -Wall -fopenmp CURE_parallel.c -std=c99 -o cure

_executable -lm
mpiexec -n <number_of_processes> ./cure_executable
```

This detailed overview covers the essential components and steps involved in the CURE clustering implementation using MPI, demonstrating how parallel processing techniques are applied to efficiently perform clustering on large datasets.


## Contributors
- **Yusuke Sugihara**
- **Joe El Khoury**

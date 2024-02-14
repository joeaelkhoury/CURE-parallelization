#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <omp.h>
#include <time.h>

// Define the data structure for a point
typedef struct {
    double x;
    double y;
} Point;

typedef struct {
    // Existing fields...
    Point* representatives;
    Point* points;
    int num_rep;
    int* pointIndices;
    int size;
    bool isActive;
    double shrink_factor;
    Point centroid;
    int* allPointIndices; 
    int allPointSize; 
    int* mergeHistory; 
    int mergeHistorySize; 
} CureCluster;

typedef struct {
    double distance;
    int rank;
} MinDistRank;

// Point calculate_centroid(CureCluster* cluster);
const int num_rep = 1;
const int shrink_factor = 0.1;

void create_mpi_point_type(MPI_Datatype* MPI_POINT) {
    const int nitems = 2; 
    int blocklengths[2] = {1, 1}; 
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[2];

    // Calculate offsets of each field
    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);

    // Create MPI struct datatype
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, MPI_POINT);
    MPI_Type_commit(MPI_POINT); 
}

// Function to read data from a file
Point* read_data(char* file_path, int* num_points, int world_rank) {
    FILE *file = fopen(file_path, "r");
    // int error_code0;
    if (!file) {
        fprintf(stderr, "Process %d unable to open file.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        
        // return NULL;
    }
    Point* points = NULL;
    double x, y;
    int count = 0;
    while (fscanf(file, "%lf,%lf\n", &x, &y) == 2) {
        if (isnan(x) || isnan(y)) {
            fprintf(stderr, "Warning: Invalid point coordinates\n");
            continue;
        }
        points = realloc(points, (count + 1) * sizeof(Point));
        points[count].x = x;
        points[count].y = y;
        count++;
    }
    fclose(file);
    *num_points = count;
    return points;
}

void distribute_data(const Point* data, int num_points, Point** local_data, int* local_num_points, int world_rank, int world_size, MPI_Datatype MPI_POINT) {
    
    // Ensure each process knows the total number of points to distribute
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate send counts and displacements
    int* sendcounts = malloc(world_size * sizeof(int));
    int* displs = malloc(world_size * sizeof(int));

    int quotient = num_points / world_size;
    int remainder = num_points % world_size;
    int sum = 0;
    for (int i = 0; i < world_size; ++i) {
        sendcounts[i] = quotient + (i < remainder ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    // Allocate buffer for local data based on calculated send count for this process
    *local_num_points = sendcounts[world_rank]; // This should match the 'rcount' in MPI_Scatterv
    *local_data = (Point*)malloc(*local_num_points * sizeof(Point));

    // Use MPI_Scatterv to distribute the data
    MPI_Scatterv(data, sendcounts, displs, MPI_POINT, *local_data, *local_num_points, MPI_POINT, 0, MPI_COMM_WORLD);

    // Debugging: Print send counts and displacements for verification
    if (world_rank == 0) {
        for (int i = 0; i < world_size; i++) {
            printf("Process %d send count: %d, displacement: %d\n", i, sendcounts[i], displs[i]);
        }
    }

    // Free allocated resources
    free(sendcounts);
    free(displs);
}

// Function to calculate the Euclidean distance between two points
double euclidean_distance(Point point1, Point point2) {
    // printf("Calculating Euclidean distance\n");
    if (isnan(point1.x) || isnan(point1.y) || isnan(point2.x) || isnan(point2.y)) {
        fprintf(stderr, "Error: Invalid point coordinates\n");
        return DBL_MAX;
    }
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}

// Function to calculate the centroid of a cluster based on its points (placeholder)
Point calculate_centroid(CureCluster* cluster) {
    printf("Entering calculate_centroid\n");
    Point centroid = {0.0, 0.0};
    double total_weight = 0.0;
    for (int i = 0; i < cluster->size; i++) {
        double weight = 1.0 / euclidean_distance(cluster->points[i], centroid);
        centroid.x += weight * cluster->points[i].x;
        centroid.y += weight * cluster->points[i].y;
        total_weight += weight;
    }
    centroid.x /= total_weight;
    centroid.y /= total_weight;
    printf("Calculated Centroid: (%f, %f)\n", centroid.x, centroid.y); 

    return centroid;
}


// Function to shrink representative points towards the cluster centroid (placeholder)
void shrink_representatives(CureCluster* cluster, Point centroid, double shrink_factor) {
    for (int i = 0; i < cluster->num_rep; i++) {
        cluster->representatives[i].x += shrink_factor * (centroid.x - cluster->representatives[i].x);
        cluster->representatives[i].y += shrink_factor * (centroid.y - cluster->representatives[i].y);
    }
}

void select_representatives(CureCluster* cluster, int num_rep) {

    // Ensure we have enough points to select from
    int reps_to_select = num_rep < cluster->size ? num_rep : cluster->size;
    cluster->num_rep = reps_to_select;
    for (int i = 0; i < reps_to_select; i++) {
        cluster->representatives[i] = cluster->points[i];
    }

}

void update_representatives(CureCluster* cluster, int num_rep) {
    // Free the old representatives
    free(cluster->representatives);

    // Allocate memory for new representatives
    cluster->representatives = malloc(num_rep * sizeof(Point));

    // Select the new representatives
    select_representatives(cluster, num_rep);
}

void find_closest_pair_representatives(CureCluster* clusters, int num_clusters, int* closest_a, int* closest_b, double* min_distance_out, int world_rank) {
    double min_distance = DBL_MAX;
    *closest_a = -1;
    *closest_b = -1;

    for (int i = 0; i < num_clusters - 1; i++) {
        for (int j = i + 1; j < num_clusters; j++) {
            double distance = euclidean_distance(clusters[i].centroid, clusters[j].centroid);
            if (distance < min_distance) {
                min_distance = distance;
                *closest_a = i;
                *closest_b = j;
            }
        }
    }

    *min_distance_out = min_distance;
    printf("Process %d: Closest pair found between clusters %d and %d with distance %f.\n", world_rank, *closest_a, *closest_b, min_distance);
}

void create_mpi_min_dist_rank_type(MPI_Datatype *MPI_MinDistRank) {
    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_INT};
    MPI_Aint offsets[2];

    offsets[0] = offsetof(MinDistRank, distance);
    offsets[1] = offsetof(MinDistRank, rank);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, MPI_MinDistRank);
    MPI_Type_commit(MPI_MinDistRank);
}

void find_global_closest_pair(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm, int* global_closest_a, int* global_closest_b, int world_rank, int world_size) {
    int closest_a = -1, closest_b = -1;
    double local_min_distance = DBL_MAX;

    // Ensure all processes have the latest data and are synchronized
    MPI_Barrier(MPI_COMM_WORLD);

    // First, find the local closest pair of clusters
    find_closest_pair_representatives(local_clusters, local_num_clusters, &closest_a, &closest_b, &local_min_distance, world_rank);

    // Structure to hold the local minimum distance and the rank of the process
    MinDistRank local_min = {local_min_distance, world_rank}, global_min;
    
    // All processes participate in the MPI_Allreduce operation to find the global minimum distance and corresponding rank
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);

    // Initialize global indices to -1, indicating no selection
    *global_closest_a = -1;
    *global_closest_b = -1;

    // If this process has the global minimum, update the global indices
    if (world_rank == global_min.rank) {
        *global_closest_a = closest_a;
        *global_closest_b = closest_b;
    }

    // Broadcast the indices of the global closest pair from the process that owns them to all other processes
    MPI_Bcast(global_closest_a, 1, MPI_INT, global_min.rank, comm);
    MPI_Bcast(global_closest_b, 1, MPI_INT, global_min.rank, comm);

}

void merge_clusters(CureCluster* clusters, int* num_clusters, int cluster_a_index, int cluster_b_index, int world_rank, MPI_Comm comm) {
    printf("Process %d: Merging clusters %d and %d.\n", world_rank, cluster_a_index, cluster_b_index);

    if (cluster_a_index < 0 || cluster_b_index < 0 || cluster_a_index >= *num_clusters || cluster_b_index >= *num_clusters || cluster_a_index == cluster_b_index) {
        fprintf(stderr, "Process %d: Invalid cluster indices %d, %d for merging.\n", world_rank, cluster_a_index, cluster_b_index);
        MPI_Abort(comm, EXIT_FAILURE);
        return;
    }

    if (cluster_a_index > cluster_b_index) {
        int temp = cluster_a_index;
        cluster_a_index = cluster_b_index;
        cluster_b_index = temp;
    }

    CureCluster* clusterA = &clusters[cluster_a_index];
    CureCluster* clusterB = &clusters[cluster_b_index];

    int mergedSize = clusterA->size + clusterB->size;
    int* mergedAllPointIndices = (int*)malloc((clusterA->allPointSize + clusterB->allPointSize) * sizeof(int));

    if (!mergedAllPointIndices) {
        fprintf(stderr, "Process %d: Failed to allocate memory for merged cluster indices.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
        return;
    }

    memcpy(mergedAllPointIndices, clusterA->allPointIndices, clusterA->allPointSize * sizeof(int));
    memcpy(mergedAllPointIndices + clusterA->allPointSize, clusterB->allPointIndices, clusterB->allPointSize * sizeof(int));

    free(clusterA->allPointIndices);
    free(clusterB->allPointIndices);

    clusterA->allPointIndices = mergedAllPointIndices;
    clusterA->allPointSize += clusterB->allPointSize;

    // If points are dynamically allocated, handle them similarly
    Point* mergedPoints = (Point*)realloc(clusterA->points, mergedSize * sizeof(Point));
    if (!mergedPoints) {
        fprintf(stderr, "Process %d: Failed to allocate memory for merged cluster points.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
        return;
    }
    memcpy(mergedPoints + clusterA->size, clusterB->points, clusterB->size * sizeof(Point));

    clusterA->points = mergedPoints;
    clusterA->size = mergedSize;

    free(clusterB->points);

    // Update centroid and representatives for the merged cluster
    clusterA->centroid = calculate_centroid(clusterA);
    select_representatives(clusterA, clusterA->num_rep);
    shrink_representatives(clusterA, clusterA->centroid, clusterA->shrink_factor);

    clusterB->isActive = false; // Mark cluster B as inactive

    // Shift the clusters array to fill the gap left by removing cluster B
    for (int i = cluster_b_index; i < *num_clusters - 1; ++i) {
        clusters[i] = clusters[i + 1];
    }
    (*num_clusters)--;

    MPI_Barrier(comm);
    printf("Process %d: Merging complete. Total clusters now: %d.\n", world_rank, *num_clusters);
}


void assign_local_labels(CureCluster* local_clusters, int local_num_clusters, int* local_labels, int local_num_points) {
    for (int i = 0; i < local_num_clusters; i++) {
        if (local_clusters[i].isActive) {
            for (int j = 0; j < local_clusters[i].size; j++) {
                int pointIndex = local_clusters[i].pointIndices[j];
                if (pointIndex >= 0 && pointIndex < local_num_points) {
                    local_labels[pointIndex] = i;
                }
            }
        }
    }
}

void calculate_sendcounts_and_displacements(int num_points, int world_size, int** sendcounts, int** displs) {
    *sendcounts = (int*)malloc(world_size * sizeof(int));
    *displs = (int*)malloc(world_size * sizeof(int));

    int quotient = num_points / world_size;
    int remainder = num_points % world_size;
    int sum = 0;
    for (int i = 0; i < world_size; ++i) {
        (*sendcounts)[i] = quotient + (i < remainder ? 1 : 0);
        (*displs)[i] = sum;
        sum += (*sendcounts)[i];
    }
}

// Function to implement the CURE clustering algorithm
int* cure_clustering(Point* data, int num_points, int n_clusters, int num_rep, double shrink_factor, MPI_Comm comm) {
    int world_size, world_rank;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);
    printf("Process %d/%d: Starting cure_clustering\n", world_rank, world_size);

    // Define MPI datatype for Point
    MPI_Datatype MPI_POINT;
    create_mpi_point_type(&MPI_POINT); // Assuming this function is defined elsewhere

    MPI_Type_contiguous(2, MPI_DOUBLE, &MPI_POINT);
    MPI_Type_commit(&MPI_POINT);
    printf("Process %d: MPI_POINT datatype defined and committed\n", world_rank);
    int* sendcounts = NULL;
    int* displs = NULL;
    calculate_sendcounts_and_displacements(num_points, world_size, &sendcounts, &displs);

    Point* local_data = NULL;
    int local_num_points = 0;
    // Use distribute_data to distribute points among processes
    distribute_data(data, num_points, &local_data, &local_num_points, world_rank, world_size, MPI_POINT);

    printf("Process %d: Points scattered to processes\n", world_rank); 

    // Initialize clusters for each process
    int num_clusters = sendcounts[world_rank];
    CureCluster* clusters = (CureCluster*)malloc(num_clusters * sizeof(CureCluster));

    for (int i = 0; i < num_clusters; i++) {
        clusters[i].points = malloc(sizeof(Point)); // Allocate space for a single point
        clusters[i].points[0] = local_data[i]; // Assign the point
        clusters[i].size = 1; // Initial size is 1
        clusters[i].representatives = malloc(num_rep * sizeof(Point)); // Allocate space for representatives
        clusters[i].num_rep = num_rep; // Set the number of representatives
        clusters[i].centroid = local_data[i]; // Initial centroid is the point itself
        clusters[i].shrink_factor = shrink_factor; // Set the shrink factor
        clusters[i].pointIndices = malloc(sizeof(int)); // Allocate space for a single point index
        clusters[i].pointIndices[0] = i; // The index of the point in the original dataset
        clusters[i].isActive = true; // Initially, all clusters are active
        clusters[i].allPointSize = 1; // Initialize allPointSize
        clusters[i].allPointIndices = malloc(sizeof(int)); // Initially points to itself
        clusters[i].allPointIndices[0] = i;
        // Initialize mergeHistory with the cluster's own index
        clusters[i].mergeHistory = malloc(sizeof(int));
        clusters[i].mergeHistory[0] = i;
        clusters[i].mergeHistorySize = 1;
    }

    printf("Process %d: Clusters initialized\n", world_rank); 
    printf("Process %d: Clusters initialized, total clusters: %d\n", world_rank, num_clusters);

    // Before deciding whether to merge clusters in the main clustering loop
    printf("Process %d evaluating if it should merge clusters.\n", world_rank);
    MPI_Barrier(MPI_COMM_WORLD); // Sync point

    // Main clustering loop
    while (num_clusters > n_clusters) {
        int closest_a = -1, closest_b = -1;
        // double min_distance = DBL_MAX;

        // Find the closest pair of clusters among local clusters
        find_global_closest_pair(clusters, num_clusters, comm, &closest_a, &closest_b, world_rank, world_size);

        // If a pair to merge is found
        if (closest_a != -1 && closest_b != -1) {
            // Merge clusters
            merge_clusters(clusters, &num_clusters, closest_a, closest_b, world_rank, comm);

        }

        // Synchronize all processes to ensure a consistent view of the cluster count
        MPI_Allreduce(MPI_IN_PLACE, &num_clusters, 1, MPI_INT, MPI_MIN, comm);
    }

    int* local_labels = (int*)malloc(sendcounts[world_rank] * sizeof(int));
    memset(local_labels, -1, sendcounts[world_rank] * sizeof(int)); // Initialize labels to -1 or a similar invalid value

    for (int i = 0; i < sendcounts[world_rank]; i++) {
        assign_local_labels(clusters, num_clusters, local_labels, sendcounts[world_rank]);
    }

    int* global_labels = NULL;
    if (world_rank == 0) {
        global_labels = (int*)malloc(num_points * sizeof(int));
    }
    MPI_Gatherv(local_labels, sendcounts[world_rank], MPI_INT, global_labels, sendcounts, displs, MPI_INT, 0, comm);

    // After the clustering is done
    if (world_rank == 0) {
        printf("Final Clusters Composition:\n");
        for (int i = 0; i < num_clusters; i++) {
            if (clusters[i].isActive) {
                printf("Cluster %d: Centroid = (%.2f, %.2f), Points Indices = [", 
                       i, clusters[i].centroid.x, clusters[i].centroid.y);
                for (int j = 0; j < clusters[i].allPointSize; j++) {
                    printf("%d ", clusters[i].allPointIndices[j]);
                }
                printf("]\n");
            }
        }
    }

    // Free dynamically allocated resources.
    for (int i = 0; i < num_clusters; i++) {
        free(clusters[i].points);
        free(clusters[i].representatives);
        free(clusters[i].pointIndices);
        free(clusters[i].allPointIndices);
        free(clusters[i].mergeHistory);
    }
    free(clusters);
    free(local_data);
    free(sendcounts);
    free(displs);
    free(local_labels);
    MPI_Type_free(&MPI_POINT);

    // Only the root process will have the meaningful global_labels
    return global_labels;
}

void visualize_clusters(CureCluster* clusters, int num_clusters) {
    for (int i = 0; i < num_clusters; i++) {
        if (!clusters[i].isActive) continue; // Skip inactive clusters
        
        printf("Cluster %d: Centroid = (%.2f, %.2f), Points in Cluster: %d\n", 
               i, clusters[i].centroid.x, clusters[i].centroid.y, clusters[i].size);
        printf("Points Indices: ");
        for (int j = 0; j < clusters[i].allPointSize; j++) {
            printf("%d ", clusters[i].allPointIndices[j]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[]) {
    int final_cluster_count;
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double start_time = MPI_Wtime();

    MPI_Datatype MPI_POINT;
    create_mpi_point_type(&MPI_POINT);

    int num_points = 0;
    Point* data = NULL;
    // Root process reads the data from file
    if (world_rank == 0) {
        // Use the DATA_FILE macro defined at compile time
        data = read_data(DATA_FILE, &num_points, world_rank);
        if (!data) {
            fprintf(stderr, "Failed to read data from %s.\n", DATA_FILE);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast the number of points to all processes
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Point* local_data = NULL;
    int local_num_points = 0;
    // Distribute data among all processes
    distribute_data(data, num_points, &local_data, &local_num_points, world_rank, world_size, MPI_POINT);

    // Free the original data array on the root process after distribution
    if (world_rank == 0) {
        free(data);
    }

    int desired_clusters = 3;
    int num_rep = 5;
    double shrink_factor = 0.5;

    CureCluster* final_clusters = NULL;

    // Each process performs clustering on its local data
    int* global_labels = cure_clustering(local_data, local_num_points, desired_clusters, num_rep, shrink_factor, MPI_COMM_WORLD);

    // Visualization and cleanup part would go here, as per your existing implementation

    MPI_Type_free(&MPI_POINT);

    // Stop the timer and print execution time on the root process
    double end_time = MPI_Wtime();
    if (world_rank == 0) {
        printf("Execution time: %f seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
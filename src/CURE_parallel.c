// Import necessary libraries
#include <mpi.h>       // For MPI functions
#include <stdio.h>     // Standard input/output
#include <stdlib.h>    // Standard library for memory allocation, process control
#include <math.h>      // Math functions
#include <float.h>     // Limits for floating-point types
#include <limits.h>    // Limits for integral types
#include <stddef.h>    // Definitions for general purpose functions
#include <string.h>    // String handling functions
#include <stdbool.h>   // Boolean type
#include <assert.h>    // Assertion functions
#include <time.h>      // Time and date functions

// Define a label for outliers in clustering
#define OUTLIER_LABEL -1

// Structure to represent a point in 2D space
typedef struct {
    double x;  // X-coordinate
    double y;  // Y-coordinate
} Point;

// Structure to represent a cluster in the CURE algorithm
typedef struct {
    Point* representatives;     // Representative points of the cluster
    Point* points;              // Points belonging to the cluster
    int num_rep;                // Number of representative points
    int* pointIndices;          // Indices of points in the cluster
    int size;                   // Number of points in the cluster
    bool isActive;              // Flag to indicate if the cluster is active
    double shrink_factor;       // Shrink factor for representative points
    Point centroid;             // Centroid of the cluster
    int* allPointIndices;       // All point indices before merging (for history)
    int allPointSize;           // Size of all points before merging
    int* mergeHistory;          // History of merges
    int mergeHistorySize;       // Number of merges
} CureCluster;

// Structure for tracking minimum distance and rank
typedef struct {
    double distance;  // Distance value
    int rank;         // Rank associated with the distance
} MinDistRank;

// Function to create an MPI datatype for a Point structure
void create_mpi_point_type(MPI_Datatype* MPI_POINT) {
    const int nitems = 2;
    int blocklengths[2] = {1, 1}; 
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE}; // Data types of Point members
    MPI_Aint offsets[2]; // Offsets of members in Point

    // Calculate offsets of x and y in the Point structure
    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);

    // Create MPI structured datatype and commit
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, MPI_POINT);
    MPI_Type_commit(MPI_POINT); 
}

// Function to read data points from a file
Point* read_data(char* file_path, int* num_points, int world_rank) {
    FILE *file = fopen(file_path, "r"); // Open file for reading
    if (!file) {
        fprintf(stderr, "Process %d unable to open file.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1); // Abort if file cannot be opened
    }

    Point* points = NULL; // Initialize pointer to Point array
    double x, y; // Temporary variables to hold point coordinates
    int count = 0; // Counter for the number of valid points read

    // Read points from file
    while (fscanf(file, "%lf,%lf\n", &x, &y) == 2) {
        if (isnan(x) || isnan(y)) {
            fprintf(stderr, "Warning: Invalid point coordinates\n");
            continue; // Skip invalid points
        }
        points = realloc(points, (count + 1) * sizeof(Point)); // Allocate/reallocate memory
        points[count].x = x; // Assign coordinates
        points[count].y = y;
        count++; // Increment counter
    }
    fclose(file); // Close the file
    *num_points = count; // Set the number of valid points read
    return points; // Return pointer to the array of points
}

// Function to calculate the Euclidean distance between two points
double euclidean_distance(Point point1, Point point2) {
    if (isnan(point1.x) || isnan(point1.y) || isnan(point2.x) || isnan(point2.y)) {
        fprintf(stderr, "Error: Invalid point coordinates\n");
        return DBL_MAX; // Return maximum double value on invalid input
    }
    // Calculate and return Euclidean distance
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}

// Function to calculate the centroid of a cluster
Point calculate_centroid(CureCluster* cluster) {
    Point centroid = {0.0, 0.0}; // Initialize centroid to (0,0)
    if (cluster->size == 0) return centroid; // Return (0,0) if cluster is empty

    // Accumulate sum of all point coordinates in the cluster
    for (int i = 0; i < cluster->size; i++) {
        centroid.x += cluster->points[i].x;
        centroid.y += cluster->points[i].y;
    }
    // Calculate average to find centroid coordinates
    centroid.x /= cluster->size;
    centroid.y /= cluster->size;
    return centroid; // Return calculated centroid
}

// Function to assign local cluster labels to each point based on their proximity to cluster centroids
void assign_local_labels(CureCluster* local_clusters, int local_num_clusters, int* local_labels, Point* local_data, int local_num_points) {
    // Allocate memory for dynamic thresholds and average distances for each cluster
    double* dynamic_thresholds = (double*)malloc(local_num_clusters * sizeof(double));
    double* average_distances = (double*)malloc(local_num_clusters * sizeof(double));
    // Check for successful memory allocation
    if (!dynamic_thresholds || !average_distances) {
        fprintf(stderr, "Failed to allocate memory for thresholds.\n");
        return;
    }
    // Calculate dynamic thresholds and average distances for each cluster
    for (int i = 0; i < local_num_clusters; i++) {
        double total_distance = 0.01; // Initialize with a small value to avoid division by zero
        // Calculate the total distance of points to the cluster centroid and find the max distance
        for (int j = 0; j < local_clusters[i].size; j++) {
            double distance = euclidean_distance(local_clusters[i].centroid, local_clusters[i].points[j]);
            total_distance += distance;
            // Update dynamic threshold if current distance is greater
            if (distance > dynamic_thresholds[i]) {
                dynamic_thresholds[i] = distance;
            }
        }
        // Calculate the average distance and set the dynamic threshold
        average_distances[i] = total_distance / local_clusters[i].size;
        dynamic_thresholds[i] = average_distances[i] * 3.0; // Threshold is 3 times the average distance
    }
    // Assign labels to points based on nearest cluster within the threshold, otherwise mark as outlier
    for (int i = 0; i < local_num_points; i++) {
        double min_distance = DBL_MAX;
        int nearest_cluster_index = -1;
        // Find the nearest cluster for each point
        for (int j = 0; j < local_num_clusters; j++) {
            double distance = euclidean_distance(local_data[i], local_clusters[j].centroid);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_cluster_index = j;
            }
        }
        // Assign the cluster label or mark as outlier
        if (nearest_cluster_index != -1 && min_distance <= dynamic_thresholds[nearest_cluster_index]) {
            local_labels[i] = nearest_cluster_index;
        } else {
            local_labels[i] = OUTLIER_LABEL;
        }
    }
    // Free allocated memory
    free(dynamic_thresholds);
    free(average_distances);
}

// Function to find the closest pair of representative points between any two clusters
void find_closest_pair_representatives(CureCluster* clusters, int num_clusters, int* closest_a, int* closest_b, double* min_distance_out, int world_rank) {
    // Validate input pointers for null values
    if (clusters == NULL || closest_a == NULL || closest_b == NULL || min_distance_out == NULL) {
        fprintf(stderr, "Error: Null pointer provided to find_closest_pair_representatives.\n");
        return;
    }

    double min_distance = DBL_MAX; // Initialize min_distance with the maximum possible value
    *closest_a = -1;
    *closest_b = -1;

    // Iterate through all pairs of clusters to find the closest pair of representatives
    for (int i = 0; i < num_clusters - 1; i++) {
        for (int j = i + 1; j < num_clusters; j++) {
            // Compare all pairs of representatives between the two clusters
            for (int ri = 0; ri < clusters[i].num_rep; ri++) {
                for (int rj = 0; rj < clusters[j].num_rep; rj++) {
                    double distance = euclidean_distance(clusters[i].representatives[ri], clusters[j].representatives[rj]);
                    // Update the closest pair if a new minimum distance is found
                    if (distance < min_distance) {
                        min_distance = distance;
                        *closest_a = i;
                        *closest_b = j;
                    }
                }
            }
        }
    }
    *min_distance_out = min_distance; // Return the minimum distance found
}

// Function to share global cluster information among all processes in a distributed computing environment
void share_global_cluster_info(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm, Point** global_centroids_out, int* total_centroids_out) {
    int world_rank, world_size;
    // Get the rank of the calling process and the size of the group in the communicator
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    // Allocate memory for local centroids
    Point* local_centroids = (Point*)malloc(local_num_clusters * sizeof(Point));
    for (int i = 0; i < local_num_clusters; i++) {
        local_centroids[i] = local_clusters[i].centroid; // Copy local centroids
    }

    // Allocate memory for storing the number of clusters per process
    int* clusters_per_process = (int*)malloc(world_size * sizeof(int));
    // Gather the number of clusters from all processes
    MPI_Allgather(&local_num_clusters, 1, MPI_INT, clusters_per_process, 1, MPI_INT, comm);

    int total_centroids = 0;
    int* displs = (int*)malloc(world_size * sizeof(int)); // Displacements for allgatherv call
    // Calculate total number of centroids and displacements for gathering data
    for (int i = 0; i < world_size; i++) {
        displs[i] = total_centroids;
        total_centroids += clusters_per_process[i];
    }

    // Allocate memory for global centroids
    Point* global_centroids = (Point*)malloc(total_centroids * sizeof(Point));
    // Gather centroids from all processes
    MPI_Allgatherv(local_centroids, local_num_clusters, MPI_DOUBLE_INT, global_centroids, clusters_per_process, displs, MPI_DOUBLE_INT, comm);

    // Free allocated memory
    free(local_centroids);
    free(clusters_per_process);
    free(displs);

    // Set output pointers
    *global_centroids_out = global_centroids;
    *total_centroids_out = total_centroids;
}


void adjust_clusters_before_local_merging(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm) {
    Point* global_centroids;
    int total_centroids;
    // Share global cluster information and get global centroids
    share_global_cluster_info(local_clusters, local_num_clusters, comm, &global_centroids, &total_centroids);

    // Iterate through local clusters to adjust centroids based on global centroids
    for (int i = 0; i < local_num_clusters; ++i) {
        double min_distance = DBL_MAX;
        int closest_global_idx = -1;
        // Find the closest global centroid to each local cluster centroid
        for (int j = 0; j < total_centroids; ++j) {
            double distance = euclidean_distance(local_clusters[i].centroid, global_centroids[j]);
            if (distance < min_distance) {
                min_distance = distance;
                closest_global_idx = j;
            }
        }
        // Adjust local cluster centroid towards the closest global centroid
        if (closest_global_idx != -1) {
            local_clusters[i].centroid.x = (local_clusters[i].centroid.x + global_centroids[closest_global_idx].x) / 2;
            local_clusters[i].centroid.y = (local_clusters[i].centroid.y + global_centroids[closest_global_idx].y) / 2;
        }
    }

    free(global_centroids); // Clean up allocated memory for global centroids
}


void adjust_representatives_towards_centroid(Point* representatives, int num_rep, Point centroid, double shrink_factor) {
    // Adjust each representative towards the centroid by the shrink factor
    for (int i = 0; i < num_rep; i++) {
        representatives[i].x = centroid.x + shrink_factor * (representatives[i].x - centroid.x);
        representatives[i].y = centroid.y + shrink_factor * (representatives[i].y - centroid.y);
    }
}

void merge_clusters_locally(CureCluster* clusters, int* num_clusters, int world_rank, MPI_Comm comm, double shrink_factor) {
    // Adjust clusters before merging based on global cluster information
    adjust_clusters_before_local_merging(clusters, *num_clusters, comm);

    // Announce the start of local merging
    printf("Process %d: Starting local merging with %d clusters.\n", world_rank, *num_clusters);

    while (*num_clusters > 3) { // Keep merging until the desired number of clusters is reached
        int merge_decision[2] = {-1, -1};
        double min_distance = DBL_MAX;

        // Find the closest pair of clusters based on their representatives' distances
        for (int i = 0; i < *num_clusters - 1; i++) {
            for (int j = i + 1; j < *num_clusters; j++) {
                if (clusters[i].isActive && clusters[j].isActive) {
                    for (int ri = 0; ri < clusters[i].num_rep; ri++) {
                        for (int rj = 0; rj < clusters[j].num_rep; rj++) {
                            double distance = euclidean_distance(clusters[i].representatives[ri], clusters[j].representatives[rj]);
                            if (distance < min_distance) {
                                min_distance = distance;
                                merge_decision[0] = i;
                                merge_decision[1] = j;
                            }
                        }
                    }
                }
            }
        }

        // Announce the closest pair to merge
        printf("Process %d: Closest pair to merge: %d and %d with distance %f.\n", world_rank, merge_decision[0], merge_decision[1], min_distance);

        // Check if a valid pair was found for merging
        if (merge_decision[0] == -1 || merge_decision[1] == -1) {
            printf("Process %d: No valid pairs to merge. Remaining clusters: %d.\n", world_rank, *num_clusters);
            break; // Exit the loop if no valid pairs are found
        } else {
            // Ensure the lower index is always merge_decision[0] for consistency
            if (merge_decision[0] > merge_decision[1]) {
                int temp = merge_decision[0];
                merge_decision[0] = merge_decision[1];
                merge_decision[1] = temp;
            }

            // Perform the merge operation for the selected clusters
            CureCluster *clusterA = &clusters[merge_decision[0]];
            CureCluster *clusterB = &clusters[merge_decision[1]];

            // Merge points from both clusters
            Point* mergedPoints = realloc(clusterA->points, (clusterA->size + clusterB->size) * sizeof(Point));
            if (!mergedPoints) {
                fprintf(stderr, "Process %d: Failed to reallocate memory for merged points.\n", world_rank);
                MPI_Abort(comm, EXIT_FAILURE);
                return;
            }
            memcpy(mergedPoints + clusterA->size, clusterB->points, clusterB->size * sizeof(Point));
            clusterA->points = mergedPoints;
            clusterA->size += clusterB->size;

            // Merge representatives from both clusters
            Point* mergedRepresentatives = (Point*)malloc((clusterA->num_rep + clusterB->num_rep) * sizeof(Point));
            if (!mergedRepresentatives) {
                fprintf(stderr, "Process %d: Failed to allocate memory for merged representatives.\n", world_rank);
                free(mergedPoints); // Free memory to prevent leaks
                MPI_Abort(comm, EXIT_FAILURE);
                return;
            }
            memcpy(mergedRepresentatives, clusterA->representatives, clusterA->num_rep * sizeof(Point));
            memcpy(mergedRepresentatives + clusterA->num_rep, clusterB->representatives, clusterB->num_rep * sizeof(Point));
            free(clusterA->representatives);
            clusterA->representatives = mergedRepresentatives;
            clusterA->num_rep += clusterB->num_rep;

            // Calculate the new centroid for the merged cluster
            clusterA->centroid = calculate_centroid(clusterA);

            // Adjust representatives towards the new centroid
            adjust_representatives_towards_centroid(clusterA->representatives, clusterA->num_rep, clusterA->centroid, shrink_factor);
            
            // Clean up the merged cluster B
            free(clusterB->points);
            free(clusterB->representatives);
            clusterB->points = NULL;
            clusterB->representatives = NULL;
            clusterB->size = 0;
            clusterB->num_rep = 0;
            clusterB->isActive = false;

            // Shift clusters to fill the gap left by the merged cluster
            for (int i = merge_decision[1]; i < *num_clusters - 1; i++) {
                clusters[i] = clusters[i + 1];
            }

            (*num_clusters)--; // Decrease the total number of clusters by one

            // Announce the completion of the merge
            printf("Process %d: Merged clusters %d and %d into one. Total active clusters now: %d.\n", world_rank, merge_decision[0], merge_decision[1], *num_clusters);
        }
    }

    // Announce the completion of local merging
    printf("Process %d: Completed local merging. Final clusters: %d.\n", world_rank, *num_clusters);
}


void global_merge_on_root(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm, double shrink_factor) {
    int world_rank, world_size;
    MPI_Comm_rank(comm, &world_rank); // Get the current process's rank
    MPI_Comm_size(comm, &world_size); // Get the total number of processes

    CureCluster* global_clusters = NULL;
    int global_num_clusters = 0;
    if (world_rank == 0) {
        // Allocate memory for global clusters on the root process
        global_clusters = (CureCluster*)malloc(world_size * local_num_clusters * sizeof(CureCluster));
        if (!global_clusters) {
            fprintf(stderr, "Memory allocation failed for global_clusters.\n");
            MPI_Abort(comm, 1);
        }
    }
    // Gather local clusters from all processes to the root process
    MPI_Gather(local_clusters, local_num_clusters * sizeof(CureCluster), MPI_BYTE, 
               global_clusters, local_num_clusters * sizeof(CureCluster), MPI_BYTE, 
               0, comm);
    if (world_rank == 0) {
        // Root process announces the gathering of all clusters
        printf("Root has gathered all clusters. Initial global_num_clusters: %d\n", world_size * local_num_clusters);
    }
    if (world_rank == 0) {
        // Root process performs final global merging
        global_num_clusters = world_size * local_num_clusters;
        while (global_num_clusters > 3) {
            double min_distance = DBL_MAX;
            int merge_idx_a = -1, merge_idx_b = -1;

            // Find the closest pair of clusters to merge
            for (int i = 0; i < global_num_clusters; i++) {
                for (int j = i + 1; j < global_num_clusters; j++) {
                    double distance = euclidean_distance(global_clusters[i].centroid, global_clusters[j].centroid);
                    if (distance < min_distance) {
                        min_distance = distance;
                        merge_idx_a = i;
                        merge_idx_b = j;
                    }
                }
            }

            // Announce the clusters selected for merging
            printf("Merging clusters %d and %d with distance %f\n", merge_idx_a, merge_idx_b, min_distance);
            if (merge_idx_a != -1 && merge_idx_b != -1) {
                // Calculate the new centroid for the merged cluster
                Point new_centroid = {
                    (global_clusters[merge_idx_a].centroid.x + global_clusters[merge_idx_b].centroid.x) / 2,
                    (global_clusters[merge_idx_a].centroid.y + global_clusters[merge_idx_b].centroid.y) / 2
                };

                // Adjust the new centroid based on the shrink factor
                new_centroid.x += shrink_factor * (new_centroid.x - global_clusters[merge_idx_a].centroid.x);
                new_centroid.y += shrink_factor * (new_centroid.y - global_clusters[merge_idx_a].centroid.y);

                // Update the centroid of the merged cluster
                global_clusters[merge_idx_a].centroid = new_centroid;
                // Shift clusters to consolidate the list after merging
                for (int i = merge_idx_b; i < global_num_clusters - 1; i++) {
                    global_clusters[i] = global_clusters[i + 1];  // Shift clusters left
                }
                global_num_clusters--;  // Decrease the total number of global clusters
                // Announce the completion of the merge
                printf("Merged clusters %d and %d. New global cluster count: %d\n", merge_idx_a, merge_idx_b, global_num_clusters);
            } else {
                // No more clusters to merge
                printf("No more clusters to merge.\n");
                break;
            }
        }
        // Announce the final global cluster count
        printf("Final global_num_clusters: %d\n", global_num_clusters);
    }

    if (world_rank == 0) {
        // Clean up allocated memory for global clusters on the root process
        free(global_clusters);
        printf("Cleaned up global_clusters.\n");
    }
}


void calculate_sendcounts_and_displacements(int num_points, int world_size, int** sendcounts, int** displs) {
    if (!sendcounts || !displs) {
        fprintf(stderr, "Null pointers provided to calculate_sendcounts_and_displacements.\n");
        return;
    }

    *sendcounts = (int*)malloc(world_size * sizeof(int));
    *displs = (int*)malloc(world_size * sizeof(int));

    if (!(*sendcounts) || !(*displs)) {
        fprintf(stderr, "Memory allocation failed in calculate_sendcounts_and_displacements.\n");
        if (*sendcounts) free(*sendcounts);
        if (*displs) free(*displs);
        *sendcounts = NULL;
        *displs = NULL;
        return;
    }

    int quotient = num_points / world_size;
    int remainder = num_points % world_size;
    int sum = 0;
    for (int i = 0; i < world_size; ++i) {
        (*sendcounts)[i] = quotient + (i < remainder ? 1 : 0);
        (*displs)[i] = sum;
        sum += (*sendcounts)[i];
    }
}


int initialize_clusters(Point* local_data, int local_data_size, CureCluster* clusters, int num_rep, double shrink_factor) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); 
    if (!local_data || !clusters) {
        fprintf(stderr, "Null pointers provided to initialize_clusters.\n");
        return -1;
    }
    int num_initial_centroids;
    Point* initial_centroids = NULL;
    if (world_rank == 0) {
        // Root process initializes centroids with local data points
        num_initial_centroids = local_data_size;
        initial_centroids = (Point*)malloc(num_initial_centroids * sizeof(Point));
        for (int i = 0; i < num_initial_centroids; i++) {
            initial_centroids[i] = local_data[i]; 
        }
    }
    // Broadcast the number of initial centroids to all processes
    MPI_Bcast(&num_initial_centroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        // Non-root processes allocate memory for initial centroids
        initial_centroids = (Point*)malloc(num_initial_centroids * sizeof(Point));
    }
    // Broadcast initial centroids to all processes
    MPI_Bcast(initial_centroids, num_initial_centroids * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < local_data_size; i++) {
        // Initialize clusters with one data point each
        clusters[i].points = (Point*)malloc(sizeof(Point));
        clusters[i].pointIndices = (int*)malloc(sizeof(int));
        if (!clusters[i].points || !clusters[i].pointIndices) {
            fprintf(stderr, "Memory allocation failed for clusters[%d] in initialize_clusters.\n", i);
            for (int j = 0; j <= i; j++) {
                if (clusters[j].points) free(clusters[j].points);
                if (clusters[j].pointIndices) free(clusters[j].pointIndices);
            }
            return -1; 
        }
        clusters[i].points[0] = local_data[i];
        clusters[i].pointIndices[0] = i;
        clusters[i].size = 1;
        clusters[i].num_rep = num_rep;
        clusters[i].shrink_factor = shrink_factor;
        clusters[i].isActive = true;
        clusters[i].allPointSize = 1;
        clusters[i].representatives = (Point*)malloc(num_rep * sizeof(Point)); // Allocate memory for representatives
        if (!clusters[i].representatives) {
            fprintf(stderr, "Memory allocation failed for representatives in clusters[%d] in initialize_clusters.\n", i);
            free(clusters[i].points);
            free(clusters[i].pointIndices);
            for (int j = 0; j < i; j++) {
                free(clusters[j].points);
                free(clusters[j].pointIndices);
                free(clusters[j].representatives);
            }
            return -1;
        }
        for (int rep = 0; rep < num_rep; rep++) {
            clusters[i].representatives[rep] = local_data[i];
        }
        clusters[i].centroid = calculate_centroid(&clusters[i]);
        if (local_data[i].x == 0 && local_data[i].y == 0) {
            printf("Process %d: Warning - Initializing cluster %d with a zero-valued point\n", world_rank, i);
        }
    }
    return local_data_size; // Return the number of initialized clusters
}

void stratified_shuffle_data(Point* data, int num_points, int num_strata_x, int num_strata_y) {
    double min_x = data[0].x, max_x = data[0].x;
    double min_y = data[0].y, max_y = data[0].y;
    // Find the minimum and maximum values for x and y coordinates
    for (int i = 1; i < num_points; i++) {
        if (data[i].x < min_x) min_x = data[i].x;
        if (data[i].x > max_x) max_x = data[i].x;
        if (data[i].y < min_y) min_y = data[i].y;
        if (data[i].y > max_y) max_y = data[i].y;
    }

    // Calculate the width and height of each stratum
    double stratum_width = (max_x - min_x) / num_strata_x;
    double stratum_height = (max_y - min_y) / num_strata_y;

    srand(time(NULL)); // Seed the random number generator

    // Assign points to strata based on their coordinates
    Point** strata = malloc(num_strata_x * num_strata_y * sizeof(Point*));
    int* strata_counts = calloc(num_strata_x * num_strata_y, sizeof(int));
    for (int i = 0; i < num_points; i++) {
        int stratum_x = (data[i].x - min_x) / stratum_width;
        int stratum_y = (data[i].y - min_y) / stratum_height;
        if (stratum_x == num_strata_x) stratum_x--; // Handle edge case
        if (stratum_y == num_strata_y) stratum_y--; // Handle edge case
        int stratum_index = stratum_y * num_strata_x + stratum_x;
        strata_counts[stratum_index]++;
    }

    // Allocate memory for each stratum based on counts
    for (int i = 0; i < num_strata_x * num_strata_y; i++) {
        strata[i] = malloc(strata_counts[i] * sizeof(Point));
        strata_counts[i] = 0; // Reset for reuse as insertion index
    }

    // Fill strata with points
    for (int i = 0; i < num_points; i++) {
        int stratum_x = (data[i].x - min_x) / stratum_width;
        int stratum_y = (data[i].y - min_y) / stratum_height;
        if (stratum_x == num_strata_x) stratum_x--; // Handle edge case
        if (stratum_y == num_strata_y) stratum_y--; // Handle edge case
        int stratum_index = stratum_y * num_strata_x + stratum_x;
        strata[stratum_index][strata_counts[stratum_index]++] = data[i];
    }

    // Shuffle each stratum individually
    for (int i = 0; i < num_strata_x * num_strata_y; i++) {
        for (int j = 0; j < strata_counts[i]; j++) {
            int k = j + rand() / (RAND_MAX / (strata_counts[i] - j) + 1);
            Point temp = strata[i][j];
            strata[i][j] = strata[i][k];
            strata[i][k] = temp;
        }
    }

    // Reassemble the shuffled data back into the original array
    int index = 0;
    for (int i = 0; i < num_strata_x * num_strata_y; i++) {
        for (int j = 0; j < strata_counts[i]; j++) {
            data[index++] = strata[i][j];
        }
        free(strata[i]); // Clean up allocated memory for each stratum
    }
    free(strata); // Clean up allocated memory for the array of strata
    free(strata_counts); // Clean up allocated memory for strata counts
}


int* cure_clustering(Point* data, int num_points, int n_clusters, int num_rep, double shrink_factor, MPI_Comm comm) {
    int world_size, world_rank;
    MPI_Comm_size(comm, &world_size); // Get the total number of processes
    MPI_Comm_rank(comm, &world_rank); // Get the current process's rank
    MPI_Datatype MPI_POINT;
    create_mpi_point_type(&MPI_POINT); // Create an MPI datatype for Point

    int num_strata_x = sqrt(world_size); // Calculate the number of strata along the x-axis
    int num_strata_y = num_strata_x; // Calculate the number of strata along the y-axis

    if (world_rank == 0) {
        // Root process shuffles data within stratified regions
        stratified_shuffle_data(data, num_points, num_strata_x, num_strata_y);
    }

    // Broadcast the number of points to all processes
    MPI_Bcast(&num_points, 1, MPI_INT, 0, comm);
    int mpi_point_size;
    MPI_Type_size(MPI_POINT, &mpi_point_size); // Get the size of the MPI_POINT datatype
    int* sendcounts = (int*)malloc(world_size * sizeof(int)); // Allocate memory for sendcounts
    int* displs = (int*)malloc(world_size * sizeof(int)); // Allocate memory for displacements
    calculate_sendcounts_and_displacements(num_points, world_size, &sendcounts, &displs); // Calculate sendcounts and displacements
    Point* local_data = (Point*)malloc(sendcounts[world_rank] * sizeof(Point)); // Allocate memory for local data
    if (!local_data) {
        fprintf(stderr, "Process %d: Failed to allocate memory for local_data.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    MPI_Barrier(comm); // Synchronize all processes
    // Scatter data points to all processes
    MPI_Scatterv(data, sendcounts, displs, MPI_POINT, local_data, sendcounts[world_rank], MPI_POINT, 0, comm);
    
    CureCluster* clusters = (CureCluster*)malloc(sendcounts[world_rank] * sizeof(CureCluster)); // Allocate memory for local clusters
    if (!clusters) {
        fprintf(stderr, "Process %d: Failed to allocate memory for clusters.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int num_local_clusters = initialize_clusters(local_data, sendcounts[world_rank], clusters, num_rep, shrink_factor); // Initialize local clusters

    // Perform local merging until the desired number of clusters is reached
    while (num_local_clusters > n_clusters) {
        int closest_a, closest_b;
        double global_min_distance = DBL_MAX;
        find_closest_pair_representatives(clusters, num_local_clusters, &closest_a, &closest_b, &global_min_distance, world_rank);
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
        merge_clusters_locally(clusters, &num_local_clusters, world_rank, comm, shrink_factor); // Perform local merging
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    } 

    // Perform additional global merging on the root process if necessary
    while (num_local_clusters > n_clusters) {
        global_merge_on_root(clusters, num_local_clusters, comm, shrink_factor);
        MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    } 

    int* local_labels = (int*)malloc(sendcounts[world_rank] * sizeof(int)); // Allocate memory for local labels
    if (!local_labels) {
        fprintf(stderr, "Process %d: Failed to allocate memory for local_labels.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
        return NULL;
    }
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes
    // Assign local labels to data points based on their nearest cluster
    assign_local_labels(clusters, num_local_clusters, local_labels, local_data, sendcounts[world_rank]);

    int* global_labels = NULL;
    if (world_rank == 0) {
        // Allocate memory for global labels on the root process
        global_labels = (int*)malloc(num_points * sizeof(int));
        if (!global_labels) {
            fprintf(stderr, "Root process: Failed to allocate memory for global_labels.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return NULL;
        }
    }

    // Gather local labels from all processes to the root process
    MPI_Gatherv(local_labels, sendcounts[world_rank], MPI_INT, global_labels, sendcounts, displs, MPI_INT, 0, comm);

    // Clean up allocated memory
    free(local_data);
    free(sendcounts);
    free(displs);
    free(local_labels);
    for (int i = 0; i < num_local_clusters; ++i) {
        free(clusters[i].points); // Free memory allocated for points in each cluster
        free(clusters[i].representatives); // Free memory allocated for representatives in each cluster
    }

    free(clusters); // Free memory allocated for clusters
    MPI_Type_free(&MPI_POINT); // Free the custom MPI datatype
    if (world_rank != 0) {
        global_labels = NULL; // Non-root processes do not hold global labels
    }
    return global_labels; // Return global labels from the root process
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Initialize the MPI environment
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size); // Get the total number of processes

    MPI_Datatype MPI_POINT;
    create_mpi_point_type(&MPI_POINT); // Create a custom MPI datatype for Point structures

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes

    double start_time = MPI_Wtime(); // Record the start time of the clustering process

    int num_points = 0;
    Point* data = NULL;
    if (world_rank == 0) {
        // Only the root process reads the full dataset
        data = read_data(DATA_FILE, &num_points, world_rank);
        if (!data) {
            fprintf(stderr, "Failed to read data.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast the number of data points to all processes
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int desired_clusters = 3, num_rep = 5; // Set the desired number of clusters and number of representatives
    double shrink_factor = 0.3; // Set the shrink factor

    // Perform CURE clustering
    int* global_labels = cure_clustering(data, num_points, desired_clusters, num_rep, shrink_factor, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes after clustering

    if (world_rank == 0) {
        // Process the clustering results on the root process
        int* clusterCounts = calloc(desired_clusters, sizeof(int)); // Allocate memory for cluster counts
        for (int i = 0; i < num_points; i++) {
            if (global_labels[i] >= 0 && global_labels[i] < desired_clusters) {
                clusterCounts[global_labels[i]]++; // Increment the count for each cluster
            }
        }
        free(clusterCounts); // Free the memory allocated for cluster counts
    }

    double end_time = MPI_Wtime(); // Record the end time of the clustering process

    if (world_rank == 0) {
        // Only the root process outputs the final clustering duration
        printf("Clustering took %f seconds.\n", end_time - start_time);
        char output_filename[256];
        char* data_file_name = getenv("DATA_FILE_NAME"); // Get the data file name from the environment variable
        if (!data_file_name) {
            fprintf(stderr, "DATA_FILE_NAME environment variable is not set.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        // Generate the output file name based on the data file name and the number of processes
        snprintf(output_filename, sizeof(output_filename), "cluster_result_%s_procs%d.csv", data_file_name, world_size);
        FILE* file = fopen(output_filename, "w"); // Open the output file for writing
        if (file == NULL) {
            fprintf(stderr, "Failed to open file '%s' for writing.\n", output_filename);
        } else {
            // Write the clustering results (data points and their assigned cluster labels) to the output file
            for (int i = 0; i < num_points; i++) {
                fprintf(file, "%f,%f,%d\n", data[i].x, data[i].y, global_labels[i]);
            }
            fclose(file); // Close the output file
            printf("Cluster labels and points saved to '%s'.\n", output_filename);
        }
    }

    if (world_rank == 0) {
        free(data); // Free the memory allocated for the dataset on the root process
    }
    if (global_labels) {
        free(global_labels); // Free the memory allocated for global labels
    }

    MPI_Type_free(&MPI_POINT); // Free the custom MPI datatype
    MPI_Finalize(); // Finalize the MPI environment
    return 0; // Exit the program
}


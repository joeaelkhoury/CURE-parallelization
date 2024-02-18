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

#define OUTLIER_LABEL -1

typedef struct {
    double x;
    double y;
} Point;

typedef struct {
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

void create_mpi_point_type(MPI_Datatype* MPI_POINT) {
    const int nitems = 2; 
    int blocklengths[2] = {1, 1}; 
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[2];

    offsets[0] = offsetof(Point, x);
    offsets[1] = offsetof(Point, y);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, MPI_POINT);
    MPI_Type_commit(MPI_POINT); 
}

Point* read_data(char* file_path, int* num_points, int world_rank) {
    FILE *file = fopen(file_path, "r");
    if (!file) {
        fprintf(stderr, "Process %d unable to open file.\n", world_rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        
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

double euclidean_distance(Point point1, Point point2) {
    if (isnan(point1.x) || isnan(point1.y) || isnan(point2.x) || isnan(point2.y)) {
        fprintf(stderr, "Error: Invalid point coordinates\n");
        return DBL_MAX;
    }
    return sqrt(pow(point1.x - point2.x, 2) + pow(point1.y - point2.y, 2));
}

Point calculate_centroid(CureCluster* cluster) {
    Point centroid = {0.0, 0.0};
    if (cluster->size == 0) return centroid; 

    for (int i = 0; i < cluster->size; i++) {
        centroid.x += cluster->points[i].x;
        centroid.y += cluster->points[i].y;
    }
    centroid.x /= cluster->size;
    centroid.y /= cluster->size;
    return centroid;
}

void assign_local_labels(CureCluster* local_clusters, int local_num_clusters, int* local_labels, Point* local_data, int local_num_points) {
    double* dynamic_thresholds = (double*)malloc(local_num_clusters * sizeof(double));
    double* average_distances = (double*)malloc(local_num_clusters * sizeof(double));
    if (!dynamic_thresholds || !average_distances) {
        fprintf(stderr, "Failed to allocate memory for thresholds.\n");
        return;
    }
    for (int i = 0; i < local_num_clusters; i++) {
        double total_distance = 0.01;
        for (int j = 0; j < local_clusters[i].size; j++) {
            double distance = euclidean_distance(local_clusters[i].centroid, local_clusters[i].points[j]);
            total_distance += distance;
            if (distance > dynamic_thresholds[i]) {
                dynamic_thresholds[i] = distance;
            }
        }
        average_distances[i] = total_distance / local_clusters[i].size;
        dynamic_thresholds[i] = average_distances[i] * 3.0; 
    }
    for (int i = 0; i < local_num_points; i++) {
        double min_distance = DBL_MAX;
        int nearest_cluster_index = -1;
        for (int j = 0; j < local_num_clusters; j++) {
            double distance = euclidean_distance(local_data[i], local_clusters[j].centroid);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_cluster_index = j;
            }
        }
        if (nearest_cluster_index != -1 && min_distance <= dynamic_thresholds[nearest_cluster_index]) {
            local_labels[i] = nearest_cluster_index;
        } else {
            local_labels[i] = OUTLIER_LABEL;
        }
    }
    free(dynamic_thresholds);
    free(average_distances);
}


void find_closest_pair_representatives(CureCluster* clusters, int num_clusters, int* closest_a, int* closest_b, double* min_distance_out, int world_rank) {
    if (clusters == NULL || closest_a == NULL || closest_b == NULL || min_distance_out == NULL) {
        fprintf(stderr, "Error: Null pointer provided to find_closest_pair_representatives.\n");
        return;
    }

    double min_distance = DBL_MAX;
    *closest_a = -1;
    *closest_b = -1;

    for (int i = 0; i < num_clusters - 1; i++) {
        for (int j = i + 1; j < num_clusters; j++) {
            for (int ri = 0; ri < clusters[i].num_rep; ri++) {
                for (int rj = 0; rj < clusters[j].num_rep; rj++) {
                    double distance = euclidean_distance(clusters[i].representatives[ri], clusters[j].representatives[rj]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        *closest_a = i;
                        *closest_b = j;
                    }
                }
            }
        }
    }
    *min_distance_out = min_distance;

    // printf("Process %d: Closest pair found between clusters %d and %d with distance %f.\n", world_rank, *closest_a, *closest_b, min_distance);
}

void share_global_cluster_info(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm, Point** global_centroids_out, int* total_centroids_out) {
    int world_rank, world_size;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    Point* local_centroids = (Point*)malloc(local_num_clusters * sizeof(Point));
    for (int i = 0; i < local_num_clusters; i++) {
        local_centroids[i] = local_clusters[i].centroid;
    }

    int* clusters_per_process = (int*)malloc(world_size * sizeof(int));
    MPI_Allgather(&local_num_clusters, 1, MPI_INT, clusters_per_process, 1, MPI_INT, comm);

    int total_centroids = 0;
    int* displs = (int*)malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++) {
        displs[i] = total_centroids;
        total_centroids += clusters_per_process[i];
    }

    Point* global_centroids = (Point*)malloc(total_centroids * sizeof(Point));
    MPI_Allgatherv(local_centroids, local_num_clusters, MPI_DOUBLE_INT, global_centroids, clusters_per_process, displs, MPI_DOUBLE_INT, comm);

    free(local_centroids);
    free(clusters_per_process);
    free(displs);

    *global_centroids_out = global_centroids;
    *total_centroids_out = total_centroids;
}

void adjust_clusters_before_local_merging(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm) {
    Point* global_centroids;
    int total_centroids;
    share_global_cluster_info(local_clusters, local_num_clusters, comm, &global_centroids, &total_centroids);

    for (int i = 0; i < local_num_clusters; ++i) {
        double min_distance = DBL_MAX;
        int closest_global_idx = -1;
        for (int j = 0; j < total_centroids; ++j) {
            double distance = euclidean_distance(local_clusters[i].centroid, global_centroids[j]);
            if (distance < min_distance) {
                min_distance = distance;
                closest_global_idx = j;
            }
        }

        if (closest_global_idx != -1) {
            local_clusters[i].centroid.x = (local_clusters[i].centroid.x + 1*global_centroids[closest_global_idx].x) / 2;
            local_clusters[i].centroid.y = (local_clusters[i].centroid.y + 1*global_centroids[closest_global_idx].y) / 2;
        }
    }

    free(global_centroids);
}

void adjust_representatives_towards_centroid(Point* representatives, int num_rep, Point centroid, double shrink_factor) {
    for (int i = 0; i < num_rep; i++) {
        representatives[i].x = centroid.x + shrink_factor * (representatives[i].x - centroid.x);
        representatives[i].y = centroid.y + shrink_factor * (representatives[i].y - centroid.y);
    }
}
void merge_clusters_locally(CureCluster* clusters, int* num_clusters, int world_rank, MPI_Comm comm, double shrink_factor) {

    adjust_clusters_before_local_merging(clusters, *num_clusters, comm);

    printf("Process %d: Starting local merging with %d clusters.\n", world_rank, *num_clusters);

    while (*num_clusters > 3) { 
        int merge_decision[2] = {-1, -1}; 
        double min_distance = DBL_MAX;

        // for (int i = 0; i < *num_clusters; i++) {
        //     for (int j = i + 1; j < *num_clusters; j++) {
        //         if (clusters[i].isActive && clusters[j].isActive) {
        //             double distance = euclidean_distance(clusters[i].centroid, clusters[j].centroid);
        //             if (distance < min_distance) {
        //                 min_distance = distance;
        //                 merge_decision[0] = i;
        //                 merge_decision[1] = j;
        //             }
        //         }
        //     }
        // }

        for (int i = 0; i < *num_clusters - 1; i++) {
            for (int j = i + 1; j < *num_clusters; j++) {
                for (int ri = 0; ri < clusters[i].num_rep; ri++) {
                    if (clusters[i].isActive && clusters[j].isActive) {
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

        printf("Process %d: Closest pair to merge: %d and %d with distance %f.\n", world_rank, merge_decision[0], merge_decision[1], min_distance);

        if (merge_decision[0] == -1 || merge_decision[1] == -1) {
            // printf("Process %d: No valid clusters to merge or decision was not made.\n", world_rank);
        } else {
            if (merge_decision[0] > merge_decision[1]) {
                int temp = merge_decision[0];
                merge_decision[0] = merge_decision[1];
                merge_decision[1] = temp;
            }
            // printf("Process %d: Proceeding with merge of clusters %d and %d.\n", world_rank, merge_decision[0], merge_decision[1]);
            if (clusters[merge_decision[0]].isActive && clusters[merge_decision[1]].isActive && merge_decision[0] != merge_decision[1]) {
                CureCluster *clusterA = &clusters[merge_decision[0]], *clusterB = &clusters[merge_decision[1]];
                Point* mergedPoints = realloc(clusterA->points, (clusterA->size + clusterB->size) * sizeof(Point));
                if (!mergedPoints) {
                    fprintf(stderr, "Process %d: Failed to reallocate memory for merged points.\n", world_rank);
                    MPI_Abort(comm, EXIT_FAILURE);
                    return;
                }
                memcpy(mergedPoints + clusterA->size, clusterB->points, clusterB->size * sizeof(Point));
                clusterA->points = mergedPoints;
                clusterA->size += clusterB->size;
                Point* mergedRepresentatives = realloc(clusterA->representatives, (clusterA->num_rep + clusterB->num_rep) * sizeof(Point));
                if (!mergedRepresentatives) {
                    fprintf(stderr, "Process %d: Failed to reallocate memory for merged representatives.\n", world_rank);
                    free(clusterA->points);
                    MPI_Abort(comm, EXIT_FAILURE);
                    return;
                }
                memcpy(mergedRepresentatives + clusterA->num_rep, clusterB->representatives, clusterB->num_rep * sizeof(Point));
                clusterA->representatives = mergedRepresentatives;
                clusterA->num_rep += clusterB->num_rep;
                clusterA->centroid = calculate_centroid(clusterA);
                adjust_representatives_towards_centroid(clusterA->representatives, clusterA->num_rep, clusterA->centroid, shrink_factor);
                adjust_representatives_towards_centroid(clusterB->representatives, clusterB->num_rep, clusterA->centroid, shrink_factor);
                memcpy(mergedRepresentatives + clusterA->num_rep, clusterB->representatives, clusterB->num_rep * sizeof(Point));

                free(clusterB->points);
                clusterB->points = NULL;
                clusterB->size = 0;
                free(clusterB->representatives);
                clusterB->representatives = NULL;
                clusterB->num_rep = 0;
                clusterB->isActive = false;
                for (int i = merge_decision[1]; i < *num_clusters - 1; i++) {
                    clusters[i] = clusters[i + 1];
                }
                (*num_clusters)--;
                printf("Process %d: Merged into new cluster. Remaining clusters: %d.\n", world_rank, *num_clusters);
            } else {
            printf("Process %d: No valid pairs to merge. Remaining clusters: %d.\n", world_rank, *num_clusters);
            break; 
        }
        }
    }
    printf("Process %d: Completed local merging. Final clusters: %d.\n", world_rank, *num_clusters);

}

void global_merge_on_root(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm, double shrink_factor) {
    int world_rank, world_size;
    MPI_Comm_rank(comm, &world_rank);
    MPI_Comm_size(comm, &world_size);

    CureCluster* global_clusters = NULL;
    int global_num_clusters = 0;
    if (world_rank == 0) {
        global_clusters = (CureCluster*)malloc(world_size * local_num_clusters * sizeof(CureCluster));
        if (!global_clusters) {
            fprintf(stderr, "Memory allocation failed for global_clusters.\n");
            MPI_Abort(comm, 1);
        }
    }
    MPI_Gather(local_clusters, local_num_clusters * sizeof(CureCluster), MPI_BYTE, 
               global_clusters, local_num_clusters * sizeof(CureCluster), MPI_BYTE, 
               0, comm);
    if (world_rank == 0) {
        printf("Root has gathered all clusters. Initial global_num_clusters: %d\n", world_size * local_num_clusters);
    }
    if (world_rank == 0) {
        global_num_clusters = world_size * local_num_clusters;
        while (global_num_clusters > 3) {
            double min_distance = DBL_MAX;
            int merge_idx_a = -1, merge_idx_b = -1;

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
            // for (int i = 0; i < global_num_clusters - 1; i++) {
            //     for (int j = i + 1; j < global_num_clusters; j++) {
            //         for (int ri = 0; ri < clusters[i].num_rep; ri++) {
            //             for (int rj = 0; rj < clusters[j].num_rep; rj++) {
            //                 double distance = euclidean_distance(clusters[i].representatives[ri], clusters[j].representatives[rj]);
            //                 if (distance < min_distance) {
            //                     min_distance = distance;
            //                     merge_idx_a = i;
            //                     merge_idx_b = j;
            //                 }
            //             }
            //         }
            //     }
            // }

            printf("Merging clusters %d and %d with distance %f\n", merge_idx_a, merge_idx_b, min_distance);
            if (merge_idx_a != -1 && merge_idx_b != -1) {
                Point new_centroid = {
                    (global_clusters[merge_idx_a].centroid.x + global_clusters[merge_idx_b].centroid.x) / 2,
                    (global_clusters[merge_idx_a].centroid.y + global_clusters[merge_idx_b].centroid.y) / 2
                };

                new_centroid.x += shrink_factor * (new_centroid.x - global_clusters[merge_idx_a].centroid.x);
                new_centroid.y += shrink_factor * (new_centroid.y - global_clusters[merge_idx_a].centroid.y);

                global_clusters[merge_idx_a].centroid = new_centroid;
                for (int i = merge_idx_b; i < global_num_clusters - 1; i++) {
                    global_clusters[i] = global_clusters[i + 1];  // Shift clusters left
                }
                global_num_clusters--;  
                printf("Merged clusters %d and %d. New global cluster count: %d\n", merge_idx_a, merge_idx_b, global_num_clusters);
            } else {
                printf("No more clusters to merge.\n");
                break;
            }
        }
        printf("Final global_num_clusters: %d\n", global_num_clusters);
    }

    if (world_rank == 0) {
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
        num_initial_centroids = local_data_size;
        initial_centroids = (Point*)malloc(num_initial_centroids * sizeof(Point));
        for (int i = 0; i < num_initial_centroids; i++) {
            initial_centroids[i] = local_data[i]; 
        }
    }
    MPI_Bcast(&num_initial_centroids, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        initial_centroids = (Point*)malloc(num_initial_centroids * sizeof(Point));
    }
    MPI_Bcast(initial_centroids, num_initial_centroids * sizeof(Point), MPI_BYTE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < local_data_size; i++) {
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
        // printf("Process %d initializing cluster %d with point: (%f, %f), Centroid: (%f, %f)\n", 
        //     world_rank, i, local_data[i].x, local_data[i].y, clusters[i].centroid.x, clusters[i].centroid.y);
    }
    return local_data_size;
}

void stratified_shuffle_data(Point* data, int num_points, int num_strata_x, int num_strata_y) {
    double min_x = data[0].x, max_x = data[0].x;
    double min_y = data[0].y, max_y = data[0].y;
    for (int i = 1; i < num_points; i++) {
        if (data[i].x < min_x) min_x = data[i].x;
        if (data[i].x > max_x) max_x = data[i].x;
        if (data[i].y < min_y) min_y = data[i].y;
        if (data[i].y > max_y) max_y = data[i].y;
    }

    double stratum_width = (max_x - min_x) / num_strata_x;
    double stratum_height = (max_y - min_y) / num_strata_y;

    srand(time(NULL));

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
        strata_counts[i] = 0; // Reset for reuse the insertion index
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
        free(strata[i]);
    }
    free(strata);
    free(strata_counts);
}

int* cure_clustering(Point* data, int num_points, int n_clusters, int num_rep, double shrink_factor, MPI_Comm comm) {
    int world_size, world_rank;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);
    MPI_Datatype MPI_POINT;
    create_mpi_point_type(&MPI_POINT);

    int num_strata_x = sqrt(world_size);
    int num_strata_y = num_strata_x;

    if (world_rank == 0) {
        stratified_shuffle_data(data, num_points, num_strata_x, num_strata_y);
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, comm);
    int mpi_point_size;
    MPI_Type_size(MPI_POINT, &mpi_point_size);
    int* sendcounts = (int*)malloc(world_size * sizeof(int));
    int* displs = (int*)malloc(world_size * sizeof(int));
    calculate_sendcounts_and_displacements(num_points, world_size, &sendcounts, &displs);
    Point* local_data = (Point*)malloc(sendcounts[world_rank] * sizeof(Point));
    if (!local_data) {
        fprintf(stderr, "Process %d: Failed to allocate memory for local_data.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    MPI_Barrier(comm);
    MPI_Scatterv(data, sendcounts, displs, MPI_POINT, local_data, sendcounts[world_rank], MPI_POINT, 0, comm);
    


    CureCluster* clusters = (CureCluster*)malloc(sendcounts[world_rank] * sizeof(CureCluster));
    if (!clusters) {
        fprintf(stderr, "Process %d: Failed to allocate memory for clusters.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int num_local_clusters = initialize_clusters(local_data, sendcounts[world_rank], clusters, num_rep, shrink_factor);



    while (num_local_clusters > n_clusters) {
        int closest_a, closest_b;
        double global_min_distance = DBL_MAX;
        find_closest_pair_representatives(clusters, num_local_clusters, &closest_a, &closest_b, &global_min_distance, world_rank);
        MPI_Barrier(MPI_COMM_WORLD);
        merge_clusters_locally(clusters, &num_local_clusters, world_rank, comm, shrink_factor);
        MPI_Barrier(MPI_COMM_WORLD);

    } 

    while (num_local_clusters > n_clusters) {

        global_merge_on_root(clusters, num_local_clusters, comm, shrink_factor);
        MPI_Barrier(MPI_COMM_WORLD);

    } 


    // MPI_Barrier(comm);
    int* local_labels = (int*)malloc(sendcounts[world_rank] * sizeof(int));
        if (!local_labels) {
        fprintf(stderr, "Process %d: Failed to allocate memory for local_labels.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
        return NULL;
    }
    // printf("test before assign local labels\n");
    MPI_Barrier(MPI_COMM_WORLD);
    assign_local_labels(clusters, num_local_clusters, local_labels, local_data, sendcounts[world_rank]);
    // for (int i = 0; i < sendcounts[world_rank]; i++) {
    //     printf("Process %d, Point %d, Label: %d\n", world_rank, i, local_labels[i]);
    // }
    // for (int i = 0; i < sendcounts[world_rank]; i++) {
    //     printf("Process %d, Point %d, Label: %d\n", world_rank, i, local_labels[i]);
    // }
    int* global_labels = NULL;
    if (world_rank == 0) {
        global_labels = (int*)malloc(num_points * sizeof(int));
        if (!global_labels) {
            fprintf(stderr, "Root process: Failed to allocate memory for global_labels.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return NULL;
        }
    }

    MPI_Gatherv(local_labels, sendcounts[world_rank], MPI_INT, global_labels, sendcounts, displs, MPI_INT, 0, comm);
    // if (world_rank == 0) {
    //     for (int i = 0; i < num_points; i++) {
    //         printf("Global label for point %d: %d\n", i, global_labels[i]);
    //     }
    // }

    // if (world_rank == 0) {
    //     for (int i = 0; i < num_points; i++) {
    //         printf("Global label for point %d: %d\n", i, global_labels[i]);
    //     }
    // }    
    free(local_data);
    free(sendcounts);
    free(displs);
    free(local_labels);
    for (int i = 0; i < num_local_clusters; ++i) {
        free(clusters[i].points);
        free(clusters[i].representatives);
    }

    free(clusters);
    MPI_Type_free(&MPI_POINT);
    if (world_rank != 0) {
        global_labels = NULL;
    }
    return global_labels;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Datatype MPI_POINT;
    // printf("Process %d about to call create_mpi_point_type\n", world_rank);
    create_mpi_point_type(&MPI_POINT);
    // printf("Process %d finished create_mpi_point_type\n", world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    // printf("Process %d passed MPI_Barrier after create_mpi_point_type\n", world_rank);

    double start_time = MPI_Wtime();

    int num_points = 0;
    Point* data = NULL;
    if (world_rank == 0) {
        // Use the DATA_FILE macro defined at compile time
        data = read_data(DATA_FILE, &num_points, world_rank);
        if (!data) {
            fprintf(stderr, "Failed to read data from %s.\n", DATA_FILE);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int desired_clusters = 3, num_rep = 5;
    double shrink_factor = 0.3;
    // printf("Process %d about to start clustering\n", world_rank);
    fflush(stdout);


    int* global_labels = cure_clustering(data, num_points, desired_clusters, num_rep, shrink_factor, MPI_COMM_WORLD);
    // printf("Process %d finished clustering\n", world_rank);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        int* clusterCounts = calloc(desired_clusters, sizeof(int));
        for (int i = 0; i < num_points; i++) {
            if (global_labels[i] >= 0 && global_labels[i] < desired_clusters) {
                clusterCounts[global_labels[i]]++;
            }
        }
        // for (int i = 0; i < desired_clusters; i++) {
        //     printf("Cluster %d has %d points\n", i, clusterCounts[i]);
        // }
        free(clusterCounts);
    }



    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        printf("Clustering took %f seconds.\n", end_time - start_time);
        char output_filename[256];
        char* data_file_name = getenv("DATA_FILE_NAME");
        if (!data_file_name) {
            fprintf(stderr, "DATA_FILE_NAME environment variable is not set.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        snprintf(output_filename, sizeof(output_filename), "cluster_result_%s_procs%d.csv", data_file_name, world_size);
        FILE* file = fopen(output_filename, "w");
        if (file == NULL) {
            fprintf(stderr, "Failed to open file '%s' for writing.\n", output_filename);
        } else {
            for (int i = 0; i < num_points; i++) {
                fprintf(file, "%f,%f,%d\n", data[i].x, data[i].y, global_labels[i]);
            }
            fclose(file);
            printf("Cluster labels and points saved to '%s'.\n", output_filename);
        }
    }

    if (world_rank == 0) {
        free(data);
    }
    if (global_labels) {
        free(global_labels);
    }

    MPI_Type_free(&MPI_POINT);
    MPI_Finalize();
    return 0;
}

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

const int num_rep = 1;
const int shrink_factor = 0.1;
double outlier_threshold = 0.5;

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

void distribute_data(const Point* data, int num_points, Point** local_data, int* local_num_points, int world_rank, int world_size, MPI_Datatype MPI_POINT) {
    
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* sendcounts = (int*)malloc(world_size * sizeof(int));
    int* displs = (int*)malloc(world_size * sizeof(int));

    if (!sendcounts || !displs) {
        fprintf(stderr, "Process %d: Failed to allocate memory for sendcounts or displs.\n", world_rank);
        if (sendcounts) free(sendcounts);
        if (displs) free(displs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return;
    }

    memset(sendcounts, 0, world_size * sizeof(int));
    memset(displs, 0, world_size * sizeof(int));

    int quotient = num_points / world_size;
    int remainder = num_points % world_size;
    int sum = 0;
    for (int i = 0; i < world_size; i++) {
        sendcounts[i] = quotient + (i < remainder ? 1 : 0);
        displs[i] = sum;
        sum += sendcounts[i];
    }

    *local_num_points = sendcounts[world_rank];
    *local_data = (Point*)malloc(*local_num_points * sizeof(Point));

    if (!*local_data) {
        fprintf(stderr, "Process %d: Failed to allocate memory for local_data.\n", world_rank);
        free(sendcounts);
        free(displs);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return;
    }

    memset(*local_data, 0, *local_num_points * sizeof(Point));

    MPI_Scatterv((void*)data, sendcounts, displs, MPI_POINT, *local_data, *local_num_points, MPI_POINT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        for (int i = 0; i < world_size; i++) {
            printf("Process %d send count: %d, displacement: %d\n", i, sendcounts[i], displs[i]);
        }
    }

    free(sendcounts);
    free(displs);
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

void shrink_representatives(CureCluster* cluster) {
    for (int i = 0; i < cluster->num_rep; i++) {
        cluster->representatives[i].x = cluster->centroid.x + shrink_factor * (cluster->representatives[i].x - cluster->centroid.x);
        cluster->representatives[i].y = cluster->centroid.y + shrink_factor * (cluster->representatives[i].y - cluster->centroid.y);
    }
}

void select_representatives(CureCluster* cluster, int num_rep) {
    if (cluster->size == 0 || num_rep <= 0) return; 

    cluster->representatives = malloc(num_rep * sizeof(Point));
    if (cluster->representatives == NULL) return; 
    cluster->num_rep = num_rep;

    double* maxDistances = malloc(cluster->size * sizeof(double));
    if (maxDistances == NULL) return; 
    for (int i = 0; i < cluster->size; i++) {
        maxDistances[i] = DBL_MAX;
    }

    cluster->representatives[0] = cluster->points[0];
    for (int i = 0; i < cluster->size; i++) {
        double dist = euclidean_distance(cluster->points[0], cluster->points[i]);
        maxDistances[i] = dist;
    }

    for (int rep = 1; rep < num_rep; rep++) {
        int farthestPointIndex = -1;
        double maxDistance = -1;
        for (int i = 0; i < cluster->size; i++) {
            double distToClosestRep = euclidean_distance(cluster->representatives[rep - 1], cluster->points[i]);
            if (distToClosestRep < maxDistances[i]) {
                maxDistances[i] = distToClosestRep;
            }
            if (maxDistances[i] > maxDistance) {
                maxDistance = maxDistances[i];
                farthestPointIndex = i;
            }
        }

        if (farthestPointIndex != -1) {
            cluster->representatives[rep] = cluster->points[farthestPointIndex];
        }
    }

    free(maxDistances);
}

void update_representatives(CureCluster* cluster, int num_rep) {
    free(cluster->representatives);
    cluster->representatives = malloc(num_rep * sizeof(Point));
    select_representatives(cluster, num_rep);
}


void assign_local_labels(CureCluster* local_clusters, int local_num_clusters, int* local_labels, Point* local_data, int local_num_points, double outlier_threshold) {
    for (int i = 0; i < local_num_points; i++) {
        local_labels[i] = -1; // -1 indicates an outlier
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

        if (min_distance <= outlier_threshold) {
            local_labels[i] = nearest_cluster_index;
        }
    }

    printf("Process labels after assignment:\n");
    for (int i = 0; i < local_num_points; i++) {
        printf("%d\n", local_labels[i]);
    }
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

    printf("Process %d: Closest pair found between clusters %d and %d with distance %f.\n", world_rank, *closest_a, *closest_b, min_distance);
}


void find_global_closest_pair(CureCluster* local_clusters, int local_num_clusters, MPI_Comm comm, int* global_closest_a, int* global_closest_b, int world_rank, int world_size) {
    printf("Process %d: Starting find_global_closest_pair.\n", world_rank);
    if (local_clusters == NULL || global_closest_a == NULL || global_closest_b == NULL || local_num_clusters <= 0) {
        fprintf(stderr, "Error: Invalid input provided to find_global_closest_pair.\n");
        MPI_Abort(comm, EXIT_FAILURE); 
        return;
    }
    
    int closest_a = -1, closest_b = -1;
    double local_min_distance = DBL_MAX;
    find_closest_pair_representatives(local_clusters, local_num_clusters, &closest_a, &closest_b, &local_min_distance, world_rank);
    printf("Process %d: Local closest pair found between clusters %d and %d with distance %f.\n", world_rank, closest_a, closest_b, local_min_distance);

    MinDistRank local_min = {local_min_distance, world_rank}, global_min;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);
    printf("Process %d: Global minimum distance found by process %d.\n", world_rank, global_min.rank);
    
    int global_indices[2];
    if (world_rank == global_min.rank) {
        global_indices[0] = closest_a;
        global_indices[1] = closest_b;
        printf("Process %d: Broadcasting global closest pair indices %d and %d.\n", world_rank, closest_a, closest_b);
    } else {
        // Initialize indices to -1 to clearly indicate they will be received
        global_indices[0] = -1;
        global_indices[1] = -1;
    }

    MPI_Bcast(global_indices, 2, MPI_INT, global_min.rank, comm);

    *global_closest_a = global_indices[0];
    *global_closest_b = global_indices[1];

    printf("Process %d: Global closest pair is between clusters %d and %d\n", world_rank, *global_closest_a, *global_closest_b);
}

void merge_clusters(CureCluster* clusters, int* num_clusters, int world_rank, MPI_Comm comm) {
    int world_size;
    MPI_Comm_size(comm, &world_size);

    int merge_decision[2] = {-1, -1}; // Initialize to invalid values

    if (world_rank == 0) {
        double min_distance = DBL_MAX;
        for (int i = 0; i < *num_clusters; i++) {
            for (int j = i + 1; j < *num_clusters; j++) {
                if (clusters[i].isActive && clusters[j].isActive) { // Consider only active clusters
                    double distance = euclidean_distance(clusters[i].centroid, clusters[j].centroid);
                    if (distance < min_distance) {
                        min_distance = distance;
                        merge_decision[0] = i;
                        merge_decision[1] = j;
                    }
                }
            }
        }
        printf("Process %d: Closest clusters to merge decided: %d and %d.\n", world_rank, merge_decision[0], merge_decision[1]);
    }
    MPI_Barrier(comm);

    MPI_Bcast(merge_decision, 2, MPI_INT, 0, comm);
    printf("Process %d received merge decision for clusters: %d and %d.\n", world_rank, merge_decision[0], merge_decision[1]);

    if (merge_decision[0] == -1 || merge_decision[1] == -1) {
        printf("Process %d: No valid clusters to merge or decision was not made.\n", world_rank);
    } else {
        if (merge_decision[0] > merge_decision[1]) {
            int temp = merge_decision[0];
            merge_decision[0] = merge_decision[1];
            merge_decision[1] = temp;
        }
        printf("Process %d: Proceeding with merge of clusters %d and %d.\n", world_rank, merge_decision[0], merge_decision[1]);
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
            // Reallocate and check memory for merged representatives
            Point* mergedRepresentatives = realloc(clusterA->representatives, (clusterA->num_rep + clusterB->num_rep) * sizeof(Point));
            if (!mergedRepresentatives) {
                fprintf(stderr, "Process %d: Failed to reallocate memory for merged representatives.\n", world_rank);
                free(clusterA->points); // Prevent memory leak
                MPI_Abort(comm, EXIT_FAILURE);
                return;
            }
            memcpy(mergedRepresentatives + clusterA->num_rep, clusterB->representatives, clusterB->num_rep * sizeof(Point));
            clusterA->representatives = mergedRepresentatives;
            clusterA->num_rep += clusterB->num_rep;
            clusterA->centroid = calculate_centroid(clusterA);
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
            (*num_clusters)--; // Decrement the total active cluster count

            printf("Process %d: Merged clusters %d and %d into one. Total active clusters now: %d.\n", world_rank, merge_decision[0], merge_decision[1], *num_clusters);
        } else {
            fprintf(stderr, "Process %d: Attempted to merge invalid or inactive clusters.\n", world_rank);
        }
    }
    MPI_Barrier(comm);
    printf("Process %d: Completed merge_clusters function.\n", world_rank);

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
        // Debug print to verify initialization
        printf("Process %d initializing cluster %d with point: (%f, %f), Centroid: (%f, %f)\n", 
            world_rank, i, local_data[i].x, local_data[i].y, clusters[i].centroid.x, clusters[i].centroid.y);
    }
    return local_data_size;
}

void visualize_clusters(CureCluster* clusters, int num_clusters, int world_rank) {
    printf("Process %d: Visualizing %d clusters.\n", world_rank, num_clusters);
    for (int i = 0; i < num_clusters; i++) {
        if (clusters[i].isActive) {
            printf("Cluster %d (Process %d): Centroid = (%f, %f)\n", i, world_rank, clusters[i].centroid.x, clusters[i].centroid.y);
            printf("Cluster %d (Process %d): Representatives = \n", i, world_rank);
            for (int j = 0; j < clusters[i].num_rep; j++) {
                printf("\t(%f, %f)\n", clusters[i].representatives[j].x, clusters[i].representatives[j].y);
            }
        }
    }
}

int* cure_clustering(Point* data, int num_points, int n_clusters, int num_rep, double shrink_factor, MPI_Comm comm) {
    int world_size, world_rank;
    double outlier_threshold = 1;
    MPI_Comm_size(comm, &world_size);
    MPI_Comm_rank(comm, &world_rank);

    MPI_Datatype MPI_POINT;
    create_mpi_point_type(&MPI_POINT);

    // Ensure all processes know the total number of points
    MPI_Bcast(&num_points, 1, MPI_INT, 0, comm);

    int mpi_point_size;
    MPI_Type_size(MPI_POINT, &mpi_point_size);
    // printf("Process %d: MPI_POINT size = %d bytes.\n", world_rank, mpi_point_size);


    int* sendcounts;
    int* displs;
    calculate_sendcounts_and_displacements(num_points, world_size, &sendcounts, &displs);

    Point* local_data = (Point*)malloc(sendcounts[world_rank] * sizeof(Point));
    if (!local_data) {
        fprintf(stderr, "Process %d: Failed to allocate memory for local_data.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (world_rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            printf("Process %d send count: %d, displacement: %d\n", i, sendcounts[i], displs[i]);
        }
    }
    MPI_Barrier(comm); // Ensure debug output is printed in order

    printf("Process %d: Receiving %d points.\n", world_rank, sendcounts[world_rank]);

    MPI_Scatterv(data, sendcounts, displs, MPI_POINT, local_data, sendcounts[world_rank], MPI_POINT, 0, comm);

    CureCluster* clusters = (CureCluster*)malloc(sendcounts[world_rank] * sizeof(CureCluster));
    if (!clusters) {
        fprintf(stderr, "Process %d: Failed to allocate memory for clusters.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
    }
    int num_local_clusters = initialize_clusters(local_data, sendcounts[world_rank], clusters, num_rep, shrink_factor);

    visualize_clusters(clusters, num_local_clusters, world_rank);

    while (num_local_clusters > n_clusters) {
        int closest_a, closest_b;
        find_global_closest_pair(clusters, num_local_clusters, comm, &closest_a, &closest_b, world_rank, world_size);
        if (closest_a != -1 && closest_b != -1) {
            merge_clusters(clusters, &num_local_clusters, world_rank, comm);
        } else {
            break;
        }
    }

    int* local_labels = (int*)malloc(sendcounts[world_rank] * sizeof(int));
        if (!local_labels) {
        fprintf(stderr, "Process %d: Failed to allocate memory for local_labels.\n", world_rank);
        MPI_Abort(comm, EXIT_FAILURE);
        return NULL;
    }

    for (int i = 0; i < sendcounts[world_rank]; ++i) {
        local_labels[i] = -1; // or appropriate initialization
    }
    assign_local_labels(clusters, num_local_clusters, local_labels, local_data, sendcounts[world_rank], outlier_threshold);
    printf("Process %d: Local label assignments:\n", world_rank);
    for (int i = 0; i < sendcounts[world_rank]; ++i) {
        printf("Process %d, Point %d, Label: %d\n", world_rank, i, local_labels[i]);
    }
    int* global_labels = NULL;
    if (world_rank == 0) {
        global_labels = (int*)malloc(num_points * sizeof(int));
        memset(global_labels, 0, num_points * sizeof(int));
        if (!global_labels) {
            fprintf(stderr, "Root process: Failed to allocate memory for global_labels.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return NULL;
        }
    }
    if (world_rank == 0) {
        for (int i = 0; i < world_size; i++) {
            printf("Receiving %d labels from process %d with displacement %d\n", sendcounts[i], i, displs[i]);
        }
    }
    MPI_Gatherv(local_labels, sendcounts[world_rank], MPI_INT, global_labels, sendcounts, displs, MPI_INT, 0, comm);
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
        free(global_labels);
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
    printf("Process %d about to call create_mpi_point_type\n", world_rank);
    create_mpi_point_type(&MPI_POINT);
    printf("Process %d finished create_mpi_point_type\n", world_rank);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d passed MPI_Barrier after create_mpi_point_type\n", world_rank);

    double start_time = MPI_Wtime();

    int num_points = 0;
    Point* data = NULL;
    if (world_rank == 0) {
        data = read_data("f.txt", &num_points, world_rank);
        if (!data) {
            fprintf(stderr, "Failed to read data.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    printf("Process %d about to call MPI_Bcast for num_points\n", world_rank);
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d finished MPI_Bcast for num_points\n", world_rank);

    Point* local_data = NULL;
    int local_num_points = 0;
    printf("Process %d about to call distribute_data\n", world_rank);
    distribute_data(data, num_points, &local_data, &local_num_points, world_rank, world_size, MPI_POINT);
    printf("Process %d finished distribute_data\n", world_rank);



    printf("Process %d received %d points:\n", world_rank, local_num_points);

    MPI_Barrier(MPI_COMM_WORLD);
    printf("Process %d passed MPI_Barrier after distribute_data\n", world_rank);



    if (world_rank == 0) {
        free(data);
    }

    int desired_clusters = 3, num_rep = 1;
    double shrink_factor = 0.1;
    printf("Process %d about to start clustering\n", world_rank);
    fflush(stdout);
    int* global_labels = cure_clustering(local_data, local_num_points, desired_clusters, num_rep, shrink_factor, MPI_COMM_WORLD);
    printf("Process %d finished clustering\n", world_rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        printf("Clustering took %f seconds.\n", end_time - start_time);
    }


    if (world_rank == 0 && global_labels) {
        FILE* file = fopen("cluster_labels0.txt", "w");
        if (file == NULL) {
            fprintf(stderr, "Failed to open file for writing.\n");
        } else {
            for (int i = 0; i < num_points; i++) {
                fprintf(file, "%d\n", global_labels[i]);
            }
            fclose(file);
            printf("Cluster labels saved to 'cluster_labels.txt'.\n");
        }
        free(global_labels);
    }

    free(local_data);
    MPI_Type_free(&MPI_POINT);
    MPI_Finalize();
    return 0;
}

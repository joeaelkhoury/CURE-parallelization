#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>


#define SOME_DEFINED_THRESHOLD 0.5
#define ANOTHER_DEFINED_THRESHOLD 0.3

// Enum to represent distance measure choices
typedef enum {
    EUCLIDEAN,
    MANHATTAN
} DistanceMeasure;


// Helper function to check if a label already exists in the unique labels array
bool contains(int *arr, int size, int value) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == value) return true;
    }
    return false;
}


// A function to compute the Euclidean distance between two points
double compute_distance(double x1, double y1, double x2, double y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// A function to compute the Manhattan distance between two points
double manhattan_distance(double x1, double y1, double x2, double y2) {
    return fabs(x1 - x2) + fabs(y1 - y2);
}

//hierarchical clustering function
int* hierarchical_clustering(double *x_coords, double *y_coords, int data_size, DistanceMeasure measure, double **representative_x, double **representative_y) {
    int desired_clusters = 10;  // Change this as per requirements
    int current_clusters = data_size;
    int *labels = (int*) malloc(data_size * sizeof(int));
    if (labels == NULL) {
        perror("Memory allocation failed for labels");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    for (int i = 0; i < data_size; i++) {
        labels[i] = i;
    }

    while (current_clusters > desired_clusters) {
        double min_distance = DBL_MAX;
        int merge_a = -1, merge_b = -1;
        
        for (int i = 0; i < data_size; i++) {
            for (int j = i + 1; j < data_size; j++) {
                if (labels[i] != labels[j]) {
                    double distance;
                    if (measure == EUCLIDEAN) {
                        distance = compute_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j]);
                    } else {  // MANHATTAN
                        distance = manhattan_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j]);
                    }
                    if (distance < min_distance) {
                        min_distance = distance;
                        merge_a = labels[i];
                        merge_b = labels[j];
                    }
                }
            }
        }

        for (int i = 0; i < data_size; i++) {
            if (labels[i] == merge_b) {
                labels[i] = merge_a;
            }
        }
        current_clusters--;
    }

    // Allocate memory for representative points
    *representative_x = (double*) malloc(desired_clusters * sizeof(double));
    if (*representative_x == NULL) {
        perror("Memory allocation failed for representative_x");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    *representative_y = (double*) malloc(desired_clusters * sizeof(double));
    if (*representative_y == NULL) {
        perror("Memory allocation failed for representative_y");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int* cluster_sizes = (int*) malloc(desired_clusters * sizeof(int));
    if (cluster_sizes == NULL) {
        perror("Memory allocation failed for cluster_sizes");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    // Initialize to 0
    for (int i = 0; i < desired_clusters; i++) {
        (*representative_x)[i] = 0;
        (*representative_y)[i] = 0;
        cluster_sizes[i] = 0;
    }

    int *unique_labels = (int*) malloc(desired_clusters * sizeof(int));
    if (unique_labels == NULL) {
        perror("Memory allocation failed for unique_labels");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    for (int i = 0; i < desired_clusters; i++) {
        unique_labels[i] = -1; // Initialize with -1 to signify "not yet found"
    }

    int found_labels = 0;
    for (int i = 0; i < data_size && found_labels < desired_clusters; i++) {
        int label = labels[i];
        if (!contains(unique_labels, found_labels, label)) {
            unique_labels[found_labels++] = label;
        }
    }

    for (int i = 0; i < data_size; i++) {
        int index = -1;
        for (int j = 0; j < desired_clusters; j++) {
            if (labels[i] == unique_labels[j]) {
                index = j;
                break;
            }
        }
        (*representative_x)[index] += x_coords[i];
        (*representative_y)[index] += y_coords[i];
        cluster_sizes[index]++;
    }
    free(unique_labels); // Return the labels for further use
}





int main(int argc, char** argv) {
    int rank, size;
    int ierr; // to handle MPI error codes
    int current_number_after_merging_and_pruning = 0;  // Initialize with zero
    int* all_labels = NULL;  // Declare this outside of the if statement
    int* portion_sizes = NULL;  // Declare the portion_sizes array

    double *x_coords, *y_coords, *subset_x, *subset_y;
    double *representative_x, *representative_y;
    int data_size, subset_size, portion_size, start_idx, end_idx;
    int *labels;
    FILE *file;

    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        perror("MPI Initialization failed");
        exit(EXIT_FAILURE);
    }

    // MPI Error Handling: Set error handlers to return errors
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    // 1. After MPI Initialization
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Reached after MPI initialization.\n");
        fflush(stdout);
    }

    double OUTLIER_THRESHOLD = 0.5;  // suggests that any representative point that is at least a distance of 0.5 away from its closest neighboring representative point is considered an outlier
    double SPLITTING_THRESHOLD = 0.3;  // suggests that if the average distance of a representative point to its neighbors is greater than 0.3, then you might consider splitting that cluster
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);



    // Only the root process reads the entire data and determines the data_size
    if (rank == 0) {
        file = fopen("/home/joe.elkhoury/project/augmented_data4.txt", "r");
        if (file == NULL) {
            perror("Failed to open the file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Count the number of lines in the file to determine data_size
        char ch;
        data_size = 0;
        while (!feof(file)) {
            ch = fgetc(file);
            if (ch == '\n') {
                data_size++;
            }
        }
        rewind(file);  // Reset file pointer to the beginning of the file

        // Allocate memory for x_coords and y_coords based on data_size
        // Add memory allocation checks after each malloc call, similar to the one done above for hierarchical_clustering
        x_coords = (double*) malloc(data_size * sizeof(double));
        if (x_coords == NULL) {
            perror("Memory allocation failed for x_coords");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        y_coords = (double*) malloc(data_size * sizeof(double));
        if (y_coords == NULL) {
            perror("Memory allocation failed for y_coords");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        // Read the data into x_coords and y_coords arrays
        for (int i = 0; i < data_size; i++) {
            fscanf(file, "%lf,%lf", &x_coords[i], &y_coords[i]);
        }

        fclose(file);
    }

    // Broadcast the data_size to all processes
    MPI_Bcast(&data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine the portion of data each process will handle
    portion_size = data_size / size;
    start_idx = rank * portion_size;
    end_idx = (rank == size - 1) ? data_size : start_idx + portion_size;

    double *local_x = (double*) malloc((end_idx - start_idx) * sizeof(double));
    double *local_y = (double*) malloc((end_idx - start_idx) * sizeof(double));

    int* sendcounts = (int*) malloc(size * sizeof(int));
    int* displs = (int*) malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        sendcounts[i] = portion_size;
        displs[i] = i * portion_size;
    }
    if (rank == size - 1) {
        sendcounts[rank] = data_size - start_idx;  // Adjust the size for the last rank
    }

    MPI_Scatterv(x_coords, sendcounts, displs, MPI_DOUBLE, local_x, end_idx - start_idx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y_coords, sendcounts, displs, MPI_DOUBLE, local_y, end_idx - start_idx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Reached after scattering data.\n");
        fflush(stdout);
    }

    free(sendcounts);
    free(displs);


    // Seed the random number generator using the rank for uniqueness
    srand(time(NULL) + rank);

    // Each process selects a subset of data points from its portion of the dataset
    subset_size = (end_idx - start_idx) / 10;
    subset_x = (double*) malloc(subset_size * sizeof(double));
    subset_y = (double*) malloc(subset_size * sizeof(double));

    // Shuffle the indices and pick the first subset_size indices
    int *indices = (int*) malloc((end_idx - start_idx) * sizeof(int));
    for (int i = 0; i < (end_idx - start_idx); i++) {
        indices[i] = i;
    }
    for (int i = (end_idx - start_idx) - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    for (int i = 0; i < subset_size; i++) {
        subset_x[i] = local_x[indices[i]];
        subset_y[i] = local_y[indices[i]];
    }
    free(indices);

    // Perform hierarchical clustering on the subset and get representative points
    DistanceMeasure chosen_measure = EUCLIDEAN;  // Or set to MANHATTAN
    labels = hierarchical_clustering(subset_x, subset_y, subset_size, chosen_measure, &representative_x, &representative_y);
    free(labels);  // Assuming labels aren't used further for now

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Reached after hierarchical clustering.\n");
    }

    // Gather the representative points from all processes
    double *all_representative_x = NULL;
    double *all_representative_y = NULL;
    if (rank == 0) {
        all_representative_x = (double*) malloc(10 * size * sizeof(double));  // 10 representative points from each process
        all_representative_y = (double*) malloc(10 * size * sizeof(double));
    }

    if (rank == 0) {
        MPI_Gather(MPI_IN_PLACE, 10, MPI_DOUBLE, all_representative_x, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(MPI_IN_PLACE, 10, MPI_DOUBLE, all_representative_y, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Gather(representative_x, 10, MPI_DOUBLE, all_representative_x, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(representative_y, 10, MPI_DOUBLE, all_representative_y, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Reached after representative points gathering.\n");
        fflush(stdout);
    }

    // Distributing the representative points across all processes for parallel processing
    // Distributing the representative points across all processes for parallel processing
    int num_representatives = 10 * size;
    int local_representatives_count = num_representatives / size;
    double *local_representative_x = (double*) malloc(local_representatives_count * sizeof(double));
    double *local_representative_y = (double*) malloc(local_representatives_count * sizeof(double));

    // Ensure that the memory for all_representative_x and all_representative_y is allocated on the root before the gather
    if (rank == 0) {
        all_representative_x = (double*) realloc(all_representative_x, num_representatives * sizeof(double));
        all_representative_y = (double*) realloc(all_representative_y, num_representatives * sizeof(double));
    }

    MPI_Scatter(all_representative_x, local_representatives_count, MPI_DOUBLE, local_representative_x, local_representatives_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(all_representative_y, local_representatives_count, MPI_DOUBLE, local_representative_y, local_representatives_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Further processing on representative points (e.g., merging, pruning) if necessary
    if (rank == 0) {
        double merge_threshold = OUTLIER_THRESHOLD;  // This value can be adjusted
        int num_representatives = 10 * size;  // Total number of representative points from all processes

        current_number_after_merging_and_pruning = num_representatives;

        // Merging
        for (int i = 0; i < num_representatives; i++) {
            for (int j = i + 1; j < num_representatives; j++) {
                double dist = compute_distance(all_representative_x[i], all_representative_y[i], all_representative_x[j], all_representative_y[j]);
                if (dist < merge_threshold) {
                    // Merge the two representative points by averaging their coordinates
                    all_representative_x[i] = (all_representative_x[i] + all_representative_x[j]) / 2.0;
                    all_representative_y[i] = (all_representative_y[i] + all_representative_y[j]) / 2.0;

                    // Move the last representative to the current j-th position and decrease the total count
                    all_representative_x[j] = all_representative_x[num_representatives - 1];
                    all_representative_y[j] = all_representative_y[num_representatives - 1];
                    num_representatives--;
                    j--;  // Re-check with the new representative moved to the j-th position
                }
            }
        }

        // Pruning (For demonstration: Let's prune points that are still very close after merging)
        for (int i = 0; i < num_representatives; i++) {
            for (int j = i + 1; j < num_representatives; j++) {
                double dist = compute_distance(all_representative_x[i], all_representative_y[i], all_representative_x[j], all_representative_y[j]);
                if (dist < merge_threshold) {
                    // Move the last representative to the current j-th position and decrease the total count
                    all_representative_x[j] = all_representative_x[num_representatives - 1];
                    all_representative_y[j] = all_representative_y[num_representatives - 1];
                    num_representatives--;
                    j--;  // Re-check with the new representative moved to the j-th position
                }
            }
        }

        current_number_after_merging_and_pruning = num_representatives;

        // Refinement by exchanging outliers among aggregated sets
        for (int i = 0; i < num_representatives; i++) {
            int closest_rep_idx = -1;
            double min_distance = DBL_MAX;
            for (int j = 0; j < num_representatives; j++) {
                if (i != j) {
                    double distance = compute_distance(all_representative_x[i], all_representative_y[i], all_representative_x[j], all_representative_y[j]);
                    if (distance < min_distance) {
                        min_distance = distance;
                        closest_rep_idx = j;
                    }
                }
            }

            double OUTLIER_THRESHOLD = 2;  // Define this value
            if (min_distance > OUTLIER_THRESHOLD) {
                // Reassign the outlier to the closest representative cluster
                all_representative_x[i] = (all_representative_x[i] + all_representative_x[closest_rep_idx]) / 2.0;
                all_representative_y[i] = (all_representative_y[i] + all_representative_y[closest_rep_idx]) / 2.0;
            }
        }

        // Splitting into new clusters if necessary
        double SPLITTING_THRESHOLD = ANOTHER_DEFINED_THRESHOLD;  // Define this value
        for (int i = 0; i < num_representatives; i++) {
            double avg_distance = 0.0;
            int neighbors = 0;
            for (int j = 0; j < num_representatives; j++) {
                if (i != j) {
                    double distance = compute_distance(all_representative_x[i], all_representative_y[i], all_representative_x[j], all_representative_y[j]);
                    if (distance < OUTLIER_THRESHOLD) {
                        avg_distance += distance;
                        neighbors++;
                    }
                }
            }
            avg_distance /= neighbors;

            if (avg_distance > SPLITTING_THRESHOLD) {
                if (num_representatives >= 10 * size) {
                    // Reallocate memory
                    size_t new_size = (num_representatives + 10) * sizeof(double);  // Increase by 10 or any other number
                    all_representative_x = (double*) realloc(all_representative_x, new_size);
                    all_representative_y = (double*) realloc(all_representative_y, new_size);
                }

                // Logic to split the cluster, for simplicity, we can duplicate the representative point with a slight offset
                all_representative_x[num_representatives] = all_representative_x[i] + 0.05;  // Small offset
                all_representative_y[num_representatives] = all_representative_y[i] + 0.05;  // Small offset
                num_representatives++;
            }

        }
    }

    // After refining, gather the data back to the root process
    MPI_Gather(local_representative_x, local_representatives_count, MPI_DOUBLE, all_representative_x, local_representatives_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_representative_y, local_representatives_count, MPI_DOUBLE, all_representative_y, local_representatives_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast the final representative points to all processes
    if (rank == 0) {
        // num_representatives = 10 * size;  // This might be changed due to merging, pruning, etc.
        num_representatives = current_number_after_merging_and_pruning;

    }
    MPI_Bcast(&num_representatives, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        all_representative_x = (double*) malloc(num_representatives * sizeof(double));
        all_representative_y = (double*) malloc(num_representatives * sizeof(double));
    }
    MPI_Bcast(all_representative_x, num_representatives, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_representative_y, num_representatives, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Final Membership Assignment
    int* final_labels = (int*) malloc((end_idx - start_idx) * sizeof(int));

    for (int i = 0; i < (end_idx - start_idx); i++) {
        double min_dist = DBL_MAX;
        int closest_rep = -1;
        for (int j = 0; j < num_representatives; j++) {
            double dist = compute_distance(local_x[i], local_y[i], all_representative_x[j], all_representative_y[j]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_rep = j;
            }
        }
        final_labels[i] = closest_rep;
    }
  
    // Before the MPI_Gatherv call, allocate memory for portion_sizes and displs
    if (rank == 0) {
        portion_sizes = (int*) malloc(size * sizeof(int));
        displs = (int*) malloc(size * sizeof(int));
        for (int i = 0; i < size; i++) {
            portion_sizes[i] = i < size - 1 ? portion_size : data_size - i * portion_size;
            displs[i] = i * portion_size;
        }
    }
    if (rank == 0) {
        all_labels = (int*) malloc(data_size * sizeof(int));
    }

    MPI_Gatherv(final_labels, end_idx - start_idx, MPI_INT, all_labels, portion_sizes, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // Write the output to a file on the root process
    if (rank == 0) {
        FILE* outfile = fopen("cluster_outputwith1.txt", "w");
        if (outfile == NULL) {
            perror("Failed to open output file");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        // Write the header
        fprintf(outfile, "DataPointIndex,XCoordinate,YCoordinate,ClusterNumber\n");
        
        for (int i = 0; i < data_size; i++) {
            fprintf(outfile, "%d,%lf,%lf,%d\n", i, x_coords[i], y_coords[i], all_labels[i]);
        }
        
        fclose(outfile);
    }
    if (rank == 0) {
        free(portion_sizes);
        free(displs);
        free(all_labels);
    }
    // Cleanup
    free(subset_x);
    free(subset_y);
    free(local_x);
    free(local_y);
    free(representative_x);
    free(representative_y);
    free(local_representative_x);
    free(local_representative_y);
    free(final_labels);
    if (rank == 0) {
        free(x_coords);
        free(y_coords);
        free(all_representative_x);
        free(all_representative_y);
    }

    if (rank == 0) {
        double end_time = MPI_Wtime();  // Stop the timer
        printf("Elapsed time: %f seconds\n", end_time - start_time);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Reached before finalization.\n");
        fflush(stdout);
    }

    MPI_Finalize();

    return 0;
}

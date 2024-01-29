---
header-includes:
  - \usepackage[margin=1in]{geometry}
---

# HPC Project - Parallelization of CURE Clustering Algorithm
- Author: __Joe El Khoury__,__Yusuke Sugihara__
- Date: 2024-xx-xx

# Introduction

## The CURE(Clustering Using REpresentatives) Algorithm
The CURE algorithm is a hierarchical clustering algorithm that is designed to overcome the limitations of the K-Means algorithm. The main process of the CURE algorithm is as follows:

1. Initialize:
   - Randomly select a subset of points from the dataset.
   - Set the desired number of clusters, k.
   - Set the number of representative points, c, for each cluster.
   - Set the shrinking factor, α.

2. Hierarchical Agglomerative Clustering on the Sample:
   - Start with each point in the sample as an individual cluster.
   - While the number of clusters > k:
     - Find the pair of clusters that are closest to each other (using a defined distance measure).
     - Merge the closest pair of clusters into a single cluster.

3. Selecting Representative Points:
   - For each of the k clusters:
     - Identify c farthest points within the cluster. These are the representative points.
     - Move each representative point towards the centroid of the cluster by a factor of α.

4. Assigning Remaining Points:
   - For each point in the original dataset not in the sample:
     - Find the nearest representative point from all k clusters.
     - Assign the point to the cluster of the nearest representative.

5. Optional: Refinement:
   - Optionally, the representative points and point assignments can be adjusted iteratively until convergence is reached.

6. Output:
   - The final clusters and their representative points are the output of the CURE algorithm.


# Parallel design




# Implementation

# Performance and scalability analysis

# Conclusion

# References

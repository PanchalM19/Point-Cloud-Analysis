# Point-Cloud-Analysis

This project involves point cloud analysis with a focus on implementing robust algorithms for plane, sphere, and cylinder fitting, as well as the Iterated Closest Point (ICP) algorithm. Here's a brief overview of the key tasks:

## Plane Fitting:
(a) Developed a plane-fitting algorithm based on sample mean, covariance matrix, and Eigenvalues/vectors.
(b) Tested the algorithm's resilience to outliers, highlighting variations from the baseline.
(c) Introduced a RANSAC-based approach for plane fitting, comparing strengths and weaknesses.

## Sphere Fitting:
Utilize RANSAC to localize a sphere in a point cloud, addressing unknown center and radius. Generated sphere hypotheses from point cloud data.

## Cylinder Fitting:
Implemented a RANSAC-based algorithm for cylinder localization, overcoming challenges in center, orientation, and radius determination. Evaluated inliers in the segmented cloud.

## Iterated Closest Point (ICP):
(a) Devised a transformation matrix for aligning two point clouds with full correspondences.
(b) Assessed algorithm robustness to noisy data and sensitivity to point order shuffling.
(c) Implemented the ICP algorithm, ensuring effective alignment on shuffled and noisy data.

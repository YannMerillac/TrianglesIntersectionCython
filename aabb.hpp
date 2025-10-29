#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Forward declarations
struct AABB;
struct Node;

// AABB structure: defined by its minimum and maximum points
struct AABB {
    Vector3d min;
    Vector3d max;

    AABB() : min(Vector3d::Constant(numeric_limits<double>::max())),
             max(Vector3d::Constant(numeric_limits<double>::lowest())) {}

    // Method to merge this AABB with another AABB or a point
    void expand(const AABB& other) {
        min = min.cwiseMin(other.min);
        max = max.cwiseMax(other.max);
    }
    void expand(const Vector3d& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }

    void set_min_max(double x_min, double y_min, double z_min,
                     double x_max, double y_max, double z_max){
        min(0) = x_min;
        min(1) = y_min;
        min(2) = z_min;
        max(0) = x_max;
        max(1) = y_max;
        max(2) = z_max;
    }

    bool intersect(const AABB& other){
        bool overlap_axis;
        double x_start, x_end, y_start, y_end;
        for(int i=0; i<3; i++){
            x_start = min(i);
            x_end = max(i);
            y_start = other.min(i);
            y_end = other.max(i);
            overlap_axis = x_start <= y_end && y_start <= x_end;
            if(!overlap_axis){
                return false;
            }
        }
        return true;
    }
};

// Node structure for the AABB Tree
struct Node {
    AABB bbox;                     // Bounding box for this node's contained triangles
    vector<int> triangle_indices;  // Indices of triangles contained in this leaf node
    Node* left = nullptr;
    Node* right = nullptr;

    // Destructor to recursively free memory
    ~Node() {
        delete left;
        delete right;
    }
};

/**
 * @brief Computes the AABB for a single triangle.
 * @param V The vertex matrix (Eigen::MatrixX3d).
 * @param triangle_index The row index of the triangle in the F matrix.
 * @param F The face/triangle matrix (Eigen::MatrixX3i).
 * @return The AABB for the triangle.
 */
AABB computeTriangleAABB(const MatrixX3d& V, int triangle_index, const MatrixX3i& F) {
    AABB bbox;
    for (int i = 0; i < 3; ++i) {
        // Get the vertex index
        int v_idx = F(triangle_index, i);
        // Get the vertex coordinates and expand the box
        bbox.expand(V.row(v_idx).transpose());
    }
    return bbox;
}

/**
 * @brief Computes the AABB that tightly bounds a set of triangles.
 * @param V The vertex matrix.
 * @param F The face matrix.
 * @param indices The indices of the triangles to bound.
 * @return The combined AABB.
 */
AABB computeGroupAABB(const MatrixX3d& V, const MatrixX3i& F, const vector<int>& indices) {
    AABB bbox;
    for (int tri_idx : indices) {
        bbox.expand(computeTriangleAABB(V, tri_idx, F));
    }
    return bbox;
}

/**
 * @brief Recursively builds the AABB tree.
 * @param V The vertex matrix (Eigen::MatrixX3d).
 * @param F The face/triangle matrix (Eigen::MatrixX3i).
 * @param triangle_indices The indices of triangles for the current node.
 * @param leaf_threshold The maximum number of triangles in a leaf node.
 * @return The root node of the constructed subtree.
 */
Node* buildAABBTree(const MatrixX3d& V, const MatrixX3i& F, vector<int>& triangle_indices, int leaf_threshold = 10) {
    // 1. Create the new node and compute its AABB
    // cout << " **** buildAABBTree " << triangle_indices.size() << endl;
    Node* node = new Node();
    node->bbox = computeGroupAABB(V, F, triangle_indices);

    // 2. Base Case: Leaf Condition
    if (triangle_indices.size() <= leaf_threshold) {
        node->triangle_indices = triangle_indices; // Store the triangle indices in the leaf
        return node;
    }

    // 3. Subdivision Logic
    
    // a. Determine the longest axis of the node's AABB
    Vector3d extent = node->bbox.max - node->bbox.min;
    int split_axis = 0; // 0 for X, 1 for Y, 2 for Z
    if (extent(1) > extent(split_axis)) split_axis = 1;
    if (extent(2) > extent(split_axis)) split_axis = 2;

    // b. Find the split plane: the midpoint of the AABB along the longest axis
    double split_plane = node->bbox.min(split_axis) + extent(split_axis) / 2.0;

    // c. Partition the triangles
    vector<int> left_indices;
    vector<int> right_indices;

    for (int tri_idx : triangle_indices) {
        // Calculate the triangle's centroid
        Vector3d v0 = V.row(F(tri_idx, 0));
        Vector3d v1 = V.row(F(tri_idx, 1));
        Vector3d v2 = V.row(F(tri_idx, 2));
        Vector3d centroid = (v0 + v1 + v2) / 3.0;

        // Assign the triangle to the left or right group based on its centroid's coordinate
        if (centroid(split_axis) < split_plane) {
            left_indices.push_back(tri_idx);
        } else {
            right_indices.push_back(tri_idx);
        }
    }

    // d. Handle empty partitions (e.g., if all centroids are on one side)
    // To prevent infinite recursion, if one group is empty, we force a split by moving
    // the first half of the indices to one side and the rest to the other.
    if (left_indices.empty() || right_indices.empty()) {
        int half_size = triangle_indices.size() / 2;
        left_indices.assign(triangle_indices.begin(), triangle_indices.begin() + half_size);
        right_indices.assign(triangle_indices.begin() + half_size, triangle_indices.end());
    }

    // 4. Recursive Calls
    node->left = buildAABBTree(V, F, left_indices, leaf_threshold);
    node->right = buildAABBTree(V, F, right_indices, leaf_threshold);

    return node;
}

/**
 * @brief Main function to start the AABB tree construction.
 * @param V The vertex matrix (Eigen::MatrixX3d).
 * @param F The face/triangle matrix (Eigen::MatrixX3i).
 * @param leaf_threshold Maximum number of triangles in a leaf node.
 * @return The root node of the fully constructed AABB tree.
 */
Node* constructAABBTree(const MatrixX3d& V, const MatrixX3i& F, int leaf_threshold = 10) {
    if (F.rows() == 0) return nullptr;

    // Create a vector of all triangle indices (0 to F.rows() - 1)
    vector<int> all_triangle_indices(F.rows());
    iota(all_triangle_indices.begin(), all_triangle_indices.end(), 0);

    // Start the recursive build process
    return buildAABBTree(V, F, all_triangle_indices, leaf_threshold);
}

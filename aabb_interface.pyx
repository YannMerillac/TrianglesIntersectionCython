from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np
cimport numpy as cnp

cdef extern from "Eigen/Dense" namespace "Eigen":

    cdef cppclass MatrixX3d:
        MatrixX3d() except +
        MatrixX3d(int n_rows, int n_cols) except +
        double& element "operator()"(int row, int col)
        int rows()
        int cols()

    cdef cppclass MatrixX3i:
        MatrixX3i() except +
        MatrixX3i(int n_rows, int n_cols) except +
        int& element "operator()"(int row, int col)
        int rows()
        int cols()

    cdef cppclass Vector3d:
        Vector3d() except +
        double& element "operator()"(int elem)
        int size()

cdef extern from "aabb.hpp":

    cdef cppclass AABB:
        Vector3d min
        Vector3d max
        void set_min_max(double x_min, double y_min, double z_min,
                         double x_max, double y_max, double z_max)
        bint intersect(const AABB& other)
        
    struct Node:
        AABB bbox
        vector[int] triangle_indices
        Node* left
        Node* right

    Node* constructAABBTree(const MatrixX3d& V, const MatrixX3i& F, int leaf_threshold)

cdef extern from "read_off.hpp":

    struct Mesh:
        MatrixX3d vertices
        MatrixX3i triangles

    Mesh read_off_file(const string& filename) 

cdef Vector3d build_vector3d(double *v3d):
    cdef Vector3d result
    cdef double* elem
    for i in range(3):
        elem = &(result.element(i))
        elem[0] = v3d[i]
    return result

cdef cnp.ndarray read_vector3d(Vector3d in_vect):
    cdef double* elem
    cdef int i
    cdef cnp.ndarray output = np.empty(3, dtype=np.double)
    for i in range(3):
        elem = &(in_vect.element(i))
        output[i] = elem[0]
    return output

cdef MatrixX3d build_matrixxd(double *data, int n_rows):
    cdef MatrixX3d result = MatrixX3d(n_rows, 3)
    cdef double* elem
    cdef int i,j
    for i in range(n_rows):
        for j in range(3):
            elem = &(result.element(i, j))
            elem[0] = data[j + 3 * i]
    return result

def read_mesh_from_off(off_filename):
    cdef int i,j
    cdef double* v_elem
    cdef int* t_elem
    cdef string c_str_filename = off_filename.encode('UTF-8')
    cdef Mesh off_mesh = read_off_file(c_str_filename)
    cdef int n_vertices = off_mesh.vertices.rows()
    cdef int n_triangles = off_mesh.triangles.rows()
    cdef cnp.ndarray vertices = np.empty((n_vertices, 3), dtype=np.double)
    cdef cnp.ndarray triangles = np.empty((n_triangles, 3), dtype=np.int32)
    for i in range(n_vertices):
        for j in range(3):
            v_elem = &(off_mesh.vertices.element(i, j))
            vertices[i, j] = v_elem[0]
    for i in range(n_triangles):
        for j in range(3):
            t_elem = &(off_mesh.triangles.element(i, j))
            triangles[i, j] = t_elem[0]
    return vertices, triangles
    
cdef class AABBNode:

    cdef Node *node

    def __init__(self):
        self.node = NULL

    def get_triangle_indices(self):
        cdef int i
        cdef int n_triangles = self.node[0].triangle_indices.size()
        cdef cnp.ndarray output = np.empty(n_triangles, dtype=np.int32)
        for i in range(n_triangles):
            output[i] = self.node[0].triangle_indices[i]
        return output

    def get_bbox(self):
        return read_vector3d(self.node[0].bbox.min), read_vector3d(self.node[0].bbox.max)

    def get_left(self):
        if self.node[0].left is NULL:
            return None
        left_node = AABBNode()
        left_node.node = self.node[0].left
        return left_node

    def get_right(self):
        if self.node[0].right is NULL:
            return None
        right_node = AABBNode()
        right_node.node = self.node[0].right
        return right_node

    def bbox_intersect(self, limit_coords):
        cdef double x_min = limit_coords[0]
        cdef double y_min = limit_coords[1]
        cdef double z_min = limit_coords[2]
        cdef double x_max = limit_coords[3]
        cdef double y_max = limit_coords[4]
        cdef double z_max = limit_coords[5]

        cdef AABB aabb_box
        aabb_box.set_min_max(x_min, y_min, z_min, x_max, y_max, z_max)
        return self.node[0].bbox.intersect(aabb_box)
        
    def build_aabb_tree(self, double[:, :] vertices, int[:, :] triangles, int leaf_threshold):
        cdef int i,j
        cdef double *v_elem
        cdef int *t_elem
        cdef int n_triangles = triangles.shape[0]
        cdef int n_vertices = vertices.shape[0]
        cdef MatrixX3d c_vertices = MatrixX3d(n_vertices, 3)
        cdef MatrixX3i c_triangles = MatrixX3i(n_triangles, 3)
        for i in range(n_vertices):
            for j in range(3):
                v_elem = &(c_vertices.element(i, j))
                v_elem[0] = vertices[i, j]
        for i in range(n_triangles):
            for j in range(3):
                t_elem = &(c_triangles.element(i, j))
                t_elem[0] = triangles[i, j]
        self.node = constructAABBTree(c_vertices, c_triangles, leaf_threshold)
        

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector

cdef extern from "Eigen/Dense" namespace "Eigen":

    cdef cppclass MatrixXd:
        MatrixXd() except +
        MatrixXd(int n_rows, int n_cols) except +
        double& element "operator()"(int row, int col)
        int rows()
        int cols()

    cdef cppclass Vector3d:
        Vector3d() except +
        double& element "operator()"(int elem)
        int size()


cdef extern from "triangle_intersection.hpp":
    bint segment_triangle_intersection(const Vector3d &T1, const Vector3d &T2, const Vector3d &T3,
                                   const Vector3d &S1, const Vector3d &S2,
                                   Vector3d &I)
    MatrixXd segment_triangle_intersection_grad(const Vector3d &T1, const Vector3d &T2, const Vector3d &T3,
                                            const MatrixXd &T1_grad, const MatrixXd &T2_grad, const MatrixXd &T3_grad,
                                            const Vector3d &S1, const Vector3d &S2,
                                            const MatrixXd &S1_grad, const MatrixXd &S2_grad)
    bint triangle_triangle_intersection(const Vector3d &A1, const Vector3d &A2, const Vector3d &A3,
                                        const Vector3d &B1, const Vector3d &B2, const Vector3d &B3,
                                        vector[Vector3d] &intersection)


cdef Vector3d build_vector3d(double *v3d):
    cdef Vector3d result
    cdef double* elem
    for i in range(3):
        elem = &(result.element(i))
        elem[0] = v3d[i]
    return result

cdef Vector3d build_vector3d_from_xyz(double x, double y, double z):
    cdef Vector3d result
    cdef double* elem
    elem = &(result.element(0))
    elem[0] = x
    elem = &(result.element(1))
    elem[0] = y
    elem = &(result.element(2))
    elem[0] = z
    return result

cdef MatrixXd build_matrixxd(double *data, int n_rows, int n_cols):
    cdef MatrixXd result = MatrixXd(n_rows, n_cols)
    cdef double* elem
    cdef int i,j
    for i in range(n_rows):
        for j in range(n_cols):
            elem = &(result.element(i, j))
            elem[0] = data[j + n_cols * i]
    return result

def py_segment_triangle_intersection(double[:] T1, double[:] T2, double[:] T3, double[:] S1, double[:] S2):
    cdef bint intersection_flag
    cdef Vector3d vT1 = build_vector3d(&T1[0])
    cdef Vector3d vT2 = build_vector3d(&T2[0])
    cdef Vector3d vT3 = build_vector3d(&T3[0])
    cdef Vector3d vS1 = build_vector3d(&S1[0])
    cdef Vector3d vS2 = build_vector3d(&S2[0])
    cdef Vector3d vI
    intersection_flag = segment_triangle_intersection(vT1, vT2, vT3, vS1, vS2, vI)
    cdef cnp.ndarray output = np.empty(3, dtype=np.double)
    for i in range(3):
        output[i] = vI.element(i)
    return intersection_flag, output

def py_segment_triangle_intersection_grad(cnp.ndarray T1, cnp.ndarray T2, cnp.ndarray T3,
                                          cnp.ndarray T1_grad, cnp.ndarray T2_grad, cnp.ndarray T3_grad,
                                          cnp.ndarray S1, cnp.ndarray S2,
                                          cnp.ndarray S1_grad, cnp.ndarray S2_grad):
    cdef int n_dv = T1_grad.shape[1]
    cdef Vector3d vT1 = build_vector3d(<double*> T1.data)
    cdef Vector3d vT2 = build_vector3d(<double*> T2.data)
    cdef Vector3d vT3 = build_vector3d(<double*> T3.data)
    cdef Vector3d vS1 = build_vector3d(<double*> S1.data)
    cdef Vector3d vS2 = build_vector3d(<double*> S2.data)
    cdef MatrixXd vT1_grad = build_matrixxd(<double*> T1_grad.data, 3, n_dv)
    cdef MatrixXd vT2_grad = build_matrixxd(<double*> T2_grad.data, 3, n_dv)
    cdef MatrixXd vT3_grad = build_matrixxd(<double*> T3_grad.data, 3, n_dv)
    cdef MatrixXd vS1_grad = build_matrixxd(<double*> S1_grad.data, 3, n_dv)
    cdef MatrixXd vS2_grad = build_matrixxd(<double*> S2_grad.data, 3, n_dv)
    cdef MatrixXd vI_grad = segment_triangle_intersection_grad(vT1, vT2, vT3, vT1_grad, vT2_grad, vT3_grad, vS1, vS2, vS1_grad, vS2_grad)
    cdef cnp.ndarray output = np.empty((3, n_dv), dtype=np.double)
    for i in range(3):
        for j in range(n_dv):
            output[i, j] = vI_grad.element(i, j)
    return output

def py_triangle_triangle_intersection(double[:, :] tri1, double[:, :] tri2):
    cdef int i, j
    cdef Vector3d A1 = build_vector3d_from_xyz(tri1[0, 0], tri1[0, 1], tri1[0, 2])
    cdef Vector3d A2 = build_vector3d_from_xyz(tri1[1, 0], tri1[1, 1], tri1[1, 2])
    cdef Vector3d A3 = build_vector3d_from_xyz(tri1[2, 0], tri1[2, 1], tri1[2, 2])
    #
    cdef Vector3d B1 = build_vector3d_from_xyz(tri2[0, 0], tri2[0, 1], tri2[0, 2])
    cdef Vector3d B2 = build_vector3d_from_xyz(tri2[1, 0], tri2[1, 1], tri2[1, 2])
    cdef Vector3d B3 = build_vector3d_from_xyz(tri2[2, 0], tri2[2, 1], tri2[2, 2])
    # result
    cdef vector[Vector3d] result_inter
    cdef bint bool_inter = triangle_triangle_intersection(A1, A2, A3, B1, B2, B3, result_inter)
    if not bool_inter:
        return []
    # export result
    cdef int n_inter_pts = result_inter.size()
    cdef cnp.ndarray output = np.empty((n_inter_pts, 3), dtype=np.double)
    for i in range(n_inter_pts):
        for j in range(3):
            output[i, j] = result_inter[i].element(j)
    return output
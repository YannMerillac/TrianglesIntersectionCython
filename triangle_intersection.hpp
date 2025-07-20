#include <Eigen/Dense>
#include <vector>

using namespace Eigen;

bool segment_triangle_intersection(const Vector3d &T1, const Vector3d &T2, const Vector3d &T3,
                                   const Vector3d &S1, const Vector3d &S2,
                                   Vector3d &I)       
{
    constexpr float epsilon = std::numeric_limits<float>::epsilon();
    Vector3d ray_vector = S2 - S1;
    Vector3d edge1 = T2 - T1;
    Vector3d edge2 = T3 - T1;
    Vector3d ray_cross_e2 = ray_vector.cross(edge2);
    float det = edge1.dot(ray_cross_e2);

    if (det > -epsilon && det < epsilon)
        return false;    // This ray is parallel to this triangle.

    float inv_det = 1.0 / det;
    Vector3d s = S1 - T1;
    float u = inv_det * s.dot(ray_cross_e2);

    if ((u < 0 && abs(u) > epsilon) || (u > 1 && abs(u-1) > epsilon))
        return false;

    Vector3d s_cross_e1 = s.cross(edge1);
    float v = inv_det * ray_vector.dot(s_cross_e1);

    if ((v < 0 && abs(v) > epsilon) || (u + v > 1 && abs(u + v - 1) > epsilon))
        return false;

    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = inv_det * edge2.dot(s_cross_e1);

    if (t > epsilon && t < 1 + epsilon) // ray intersection
    {
        I = S1 + ray_vector * t;
        return  true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

MatrixXd segment_triangle_intersection_grad(const Vector3d &T1, const Vector3d &T2, const Vector3d &T3,
                                            const MatrixXd &T1_grad, const MatrixXd &T2_grad, const MatrixXd &T3_grad,
                                            const Vector3d &S1, const Vector3d &S2,
                                            const MatrixXd &S1_grad, const MatrixXd &S2_grad)       
{
    int n_dv = T1_grad.cols();
    MatrixXd I_grad;
    Vector3d ray_vector = S2 - S1;
    MatrixXd ray_vector_grad = S2_grad - S1_grad;
    Vector3d edge1 = T2 - T1;
    MatrixXd edge1_grad = T2_grad - T1_grad;
    Vector3d edge2 = T3 - T1;
    MatrixXd edge2_grad = T3_grad - T1_grad;
    Vector3d ray_cross_e2 = ray_vector.cross(edge2);
    float det = edge1.dot(ray_cross_e2);
    
    MatrixXd ray_cross_e2_grad(3, n_dv);
    ray_cross_e2_grad.row(0) = ray_vector_grad.row(1).array() * edge2(2) - ray_vector_grad.row(2).array() * edge2(1);
    ray_cross_e2_grad.row(1) = ray_vector_grad.row(2).array() * edge2(0) - ray_vector_grad.row(0).array() * edge2(2);
    ray_cross_e2_grad.row(2) = ray_vector_grad.row(0).array() * edge2(1) - ray_vector_grad.row(1).array() * edge2(0);

    ray_cross_e2_grad.row(0).array() += ray_vector(1) * edge2_grad.row(2).array() - ray_vector(2) * edge2_grad.row(1).array();
    ray_cross_e2_grad.row(1).array() += ray_vector(2) * edge2_grad.row(0).array() - ray_vector(0) * edge2_grad.row(2).array();
    ray_cross_e2_grad.row(2).array() += ray_vector(0) * edge2_grad.row(1).array() - ray_vector(1) * edge2_grad.row(0).array();

    VectorXd det_grad = ray_cross_e2.transpose() * edge1_grad + edge1.transpose() * ray_cross_e2_grad;

    float inv_det = 1.0 / det;
    VectorXd inv_det_grad = -det_grad / (det * det);
    Vector3d s = S1 - T1;
    MatrixXd s_grad = S1_grad - T1_grad;
    Vector3d s_cross_e1 = s.cross(edge1);
    MatrixXd s_cross_e1_grad(3, n_dv);
    s_cross_e1_grad.row(0) = s_grad.row(1).array() * edge1(2) - s_grad.row(2).array() * edge1(1);
    s_cross_e1_grad.row(1) = s_grad.row(2).array() * edge1(0) - s_grad.row(0).array() * edge1(2);
    s_cross_e1_grad.row(2) = s_grad.row(0).array() * edge1(1) - s_grad.row(1).array() * edge1(0);

    s_cross_e1_grad.row(0).array() += s(1) * edge1_grad.row(2).array() - s(2) * edge1_grad.row(1).array();
    s_cross_e1_grad.row(1).array() += s(2) * edge1_grad.row(0).array() - s(0) * edge1_grad.row(2).array();
    s_cross_e1_grad.row(2).array() += s(0) * edge1_grad.row(1).array() - s(1) * edge1_grad.row(0).array();


    float t = inv_det * edge2.dot(s_cross_e1);
    VectorXd t_grad = inv_det_grad * edge2.dot(s_cross_e1);
    t_grad += inv_det * (s_cross_e1.transpose() * edge2_grad + edge2.transpose() * s_cross_e1_grad);

    I_grad = S1_grad + ray_vector_grad * t;
    I_grad.row(0).array() += ray_vector(0) * t_grad.array();
    I_grad.row(1).array() += ray_vector(1) * t_grad.array();
    I_grad.row(2).array() += ray_vector(2) * t_grad.array();
    return I_grad;
}


int add_intersection_point(const Vector3d &I, std::vector<Vector3d> &intersection, const double& dist_tol){
    int n_inter_pts = intersection.size();
    for(int i=0; i<n_inter_pts; i++){
        if((I - intersection[i]).norm() < dist_tol){
            return n_inter_pts;
        }
    }
    intersection.push_back(I);
    return n_inter_pts + 1;
}

bool triangle_triangle_intersection(const Vector3d &A1, const Vector3d &A2, const Vector3d &A3,
                                    const Vector3d &B1, const Vector3d &B2, const Vector3d &B3,
                                    std::vector<Vector3d> &intersection)
{
    bool segment_inter;
    double dist_tol=1e-8;
    Vector3d I;

    // Edges of Triangle A
    std::array<std::pair<Vector3d, Vector3d>, 3> edges_A = {
        std::make_pair(A1, A2),
        std::make_pair(A2, A3),
        std::make_pair(A3, A1)
    };

    for (const auto& edge : edges_A){
        segment_inter = segment_triangle_intersection(B1, B2, B3, edge.first, edge.second, I);
        if(segment_inter){
            add_intersection_point(I, intersection, dist_tol);
        }
    }

    if(intersection.size() == 2){
        return true;
    }
    // Edges of Triangle B
    std::array<std::pair<Vector3d, Vector3d>, 3> edges_B = {
        std::make_pair(B1, B2),
        std::make_pair(B2, B3),
        std::make_pair(B3, B1)
    };

    for (const auto& edge : edges_B){
        segment_inter = segment_triangle_intersection(A1, A2, A3, edge.first, edge.second, I);
        if(segment_inter){
            add_intersection_point(I, intersection, dist_tol);
        }
    }

    if(intersection.size() == 2){
        return true;
    }
    return false;
}
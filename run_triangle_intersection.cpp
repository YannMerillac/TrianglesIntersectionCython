#include "triangle_intersection.hpp"
#include "iostream"
#include <Eigen/Dense>

using namespace Eigen;

int main(){
    bool intersect;
    Vector3d I;
    Vector3d T1 = {0,0,0};
    Vector3d T2 = {1,0,0};
    Vector3d T3 = {0,1,0};
    Vector3d S1 = {0.2,0.2,-1};
    Vector3d S2 = {0.2,0.2, 1};
    Vector3d S3 = {1., 1., 0.};
    intersect = segment_triangle_intersection(T1, T2, T3, S1, S2, I);
    std::cout << "Intersection between T123 and S12: " << intersect << std::endl;
    if(intersect){
        std::cout << "Result: " << I << std::endl;
    }
    // check triangle triangle intersection
    std::vector<Vector3d> tri_intersection;
    intersect = triangle_triangle_intersection(T1, T2, T3, S1, S2, S3, tri_intersection);
    std::cout << "Intersection between T123 and S123: " << intersect << std::endl;
    if(intersect){
        for (const auto& inter_pt : tri_intersection){
            std::cout << "Intersection at: " << inter_pt << std::endl;
        }
    }

}
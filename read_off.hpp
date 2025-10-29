#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

// Inclusion de la bibliothèque Eigen pour les matrices
#include <Eigen/Dense> 

// --- Data Structures ---

// Définition de l'objet Mesh utilisant les types Eigen
struct Mesh {
    // Matrice pour les coordonnées des sommets (N lignes, 3 colonnes: X, Y, Z)
    // Utilisation de 'double' (d) pour la précision des coordonnées.
    Eigen::MatrixX3d vertices; 
    
    // Matrice pour les indices des faces (triangles) (N lignes, 3 colonnes: V1, V2, V3)
    // Utilisation de 'int' (i) pour les indices.
    Eigen::MatrixX3i triangles; 
};

/**
 * @brief Reads a 3D mesh from an OFF (Object File Format) file and stores data in Eigen matrices.
 * * @param filename The path to the OFF file.
 * @return Mesh A Mesh structure populated with Eigen matrices for vertices and triangles.
 */
Mesh read_off_file(const std::string& filename) {
    std::ifstream file(filename);
    Mesh mesh;
    
    // Pour l'efficacité, nous allons lire les données dans des std::vector temporaires 
    // avant de les copier dans les matrices Eigen une fois la taille totale connue.
    std::vector<double> temp_vertices;
    std::vector<int> temp_triangle_indices;

    // 1. Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return mesh; // Return empty mesh
    }

    std::string line;
    
    // 2. Read the magic header (e.g., "OFF", "nOFF")
    if (!std::getline(file, line)) {
        std::cerr << "Error: File is empty or could not read header line." << std::endl;
        return mesh;
    }

    std::stringstream header_ss(line);
    std::string header_tag;
    header_ss >> header_tag;
    
    // Normalize header tag to uppercase for robust check
    std::transform(header_tag.begin(), header_tag.end(), header_tag.begin(), ::toupper);

    if (header_tag != "OFF" && header_tag != "NOFF") {
        std::cerr << "Error: Invalid OFF file header: " << line << std::endl;
        return mesh;
    }

    // 3. Read the vertex/face/edge count line
    long num_vertices_expected = 0;
    long num_faces_expected = 0;
    long num_edges = 0;
    
    // This loop handles empty lines and comment lines until a data line is found
    while (std::getline(file, line)) {
        if (!line.empty() && line.front() != '#') {
            break;
        }
    }

    std::stringstream counts_ss(line);

    if (!(counts_ss >> num_vertices_expected >> num_faces_expected >> num_edges)) {
        std::cerr << "Error: Failed to read vertex, face, and edge counts." << std::endl;
        return mesh;
    }

    if (num_vertices_expected <= 0 || num_faces_expected < 0) {
        std::cerr << "Error: Invalid counts detected (Vertices: " << num_vertices_expected 
                  << ", Faces: " << num_faces_expected << ")." << std::endl;
        return mesh;
    }
    
    temp_vertices.reserve(num_vertices_expected * 3);

    // 4. Read vertices
    std::cout << "Reading " << num_vertices_expected << " vertices..." << std::endl;
    for (long i = 0; i < num_vertices_expected; ++i) {
        double x, y, z;
        // Lire directement dans des doubles pour correspondre à MatrixX3d
        if (!(file >> x >> y >> z)) {
            std::cerr << "Error: Failed to read vertex " << i << " (expected " << num_vertices_expected << ")." << std::endl;
            return mesh;
        }
        temp_vertices.push_back(x);
        temp_vertices.push_back(y);
        temp_vertices.push_back(z);
    }
    
    // Redimensionner et copier les données de sommets dans la matrice Eigen
    mesh.vertices.resize(num_vertices_expected, 3);
    // Le 'Map' permet de copier efficacement les données du vector vers la matrice
    Eigen::Map<Eigen::MatrixX3d>(mesh.vertices.data(), num_vertices_expected, 3) = 
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>>(
            temp_vertices.data(), num_vertices_expected, 3);


    // 5. Read faces
    std::cout << "Reading " << num_faces_expected << " faces (only triangles will be stored)..." << std::endl;
    long num_triangles_read = 0;
    long num_non_triangular_faces = 0;
    
    for (long i = 0; i < num_faces_expected; ++i) {
        int face_vertices_count;
        if (!(file >> face_vertices_count)) {
            std::cerr << "Error: Failed to read face vertex count for face " << i << "." << std::endl;
            // Tenter de nettoyer le flux pour continuer ou retourner
            return mesh; 
        }

        if (face_vertices_count == 3) {
            // C'est un triangle, lisons les 3 indices
            int i1, i2, i3;
            if (!(file >> i1 >> i2 >> i3)) {
                std::cerr << "Error: Failed to read 3 indices for triangular face " << i << "." << std::endl;
                return mesh;
            }
            temp_triangle_indices.push_back(i1);
            temp_triangle_indices.push_back(i2);
            temp_triangle_indices.push_back(i3);
            num_triangles_read++;
        } else {
            // C'est un polygone (quad, etc.) qui ne correspond pas à MatrixX3i. Ignorer.
            num_non_triangular_faces++;
            // Lire et ignorer le reste des indices sur cette ligne pour passer à la suivante
            for (int j = 0; j < face_vertices_count; ++j) {
                int index_to_discard;
                if (!(file >> index_to_discard)) {
                     std::cerr << "Warning: Could not discard remaining indices for face " << i << "." << std::endl;
                     break;
                }
            }
        }
    }
    
    // Avertissement si des faces non-triangulaires ont été ignorées
    if (num_non_triangular_faces > 0) {
        std::cerr << "Warning: " << num_non_triangular_faces 
                  << " non-triangular faces were found and ignored to fit the MatrixX3i structure." << std::endl;
    }

    // Redimensionner et copier les données de faces dans la matrice Eigen
    if (num_triangles_read > 0) {
        mesh.triangles.resize(num_triangles_read, 3);
        Eigen::Map<Eigen::MatrixX3i>(mesh.triangles.data(), num_triangles_read, 3) = 
            Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>>(
                temp_triangle_indices.data(), num_triangles_read, 3);
    } else {
        mesh.triangles.resize(0, 3);
    }


    std::cout << "Successfully loaded mesh from " << filename << std::endl;
    return mesh;
}

// --- Utility Functions ---

void print_mesh_summary(const Mesh& mesh) {
    std::cout << "\n--- Mesh Summary ---\n";
    std::cout << "Total Vertices (MatrixX3d rows): " << mesh.vertices.rows() << "\n";
    std::cout << "Total Triangles (MatrixX3i rows): " << mesh.triangles.rows() << "\n";
    std::cout << "Columns: " << mesh.vertices.cols() << " (X, Y, Z)\n";

    if (mesh.vertices.rows() > 0) {
        std::cout << "\nFirst Vertex (Row 0):\n" 
                  << mesh.vertices.row(0) << "\n";
    }

    if (mesh.triangles.rows() > 0) {
        std::cout << "\nFirst Triangle (Row 0 - Indices V1 V2 V3):\n"
                  << mesh.triangles.row(0) << "\n";
    }
    std::cout << "----------------------\n";
}

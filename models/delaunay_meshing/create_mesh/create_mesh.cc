#include<fstream>
#include<iostream>
#include<string>
#include <unordered_set>
#include <unordered_map>
#include <iterator>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <open3d/Open3D.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_vertex_base_with_info_3<double, K>      Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K>                 Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds>                      Delaunay;

namespace std {

template<>
struct hash<Delaunay::Vertex_handle> {
    std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<const Delaunay::Vertex_handle> {
    std::size_t operator()(const Delaunay::Vertex_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<Delaunay::Cell_handle> {
    std::size_t operator()(const Delaunay::Cell_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};

template<>
struct hash<const Delaunay::Cell_handle> {
    std::size_t operator()(const Delaunay::Cell_handle& handle) const {
        return reinterpret_cast<std::size_t>(&*handle);
    }
};
}

K::Point_3 EigenToCGAL(const Eigen::Vector3d& point) {
    return K::Point_3(point(0), point(1), point(2));
}
Eigen::Vector3d CGALToEigen(const K::Point_3& point) {
    return Eigen::Vector3d(point.x(), point.y(), point.z());
}


class DelaunayMeshing
{
public:
    DelaunayMeshing(std::string& input_path,
                    std::string& input_label_file_name,
                    std::string& output_mesh_file_name);

    Delaunay trianglation_;

    std::string input_path_;
    std::string input_label_file_name_;
    std::string output_mesh_file_name_;

    std::vector<int> cell_labels_;
    std::unordered_map<const Delaunay::Cell_handle, int> cell_label_map_;

    void ReadData();
    void TransformMesh(open3d::geometry::TriangleMesh& input);
    void CreateMesh();
    void Run();
};

DelaunayMeshing::DelaunayMeshing(std::string& input_path, std::string& input_label_file_name, std::string& output_mesh_file_name) {
    input_path_ = input_path;
    input_label_file_name_ = input_label_file_name;
    output_mesh_file_name_ = output_mesh_file_name;
}

void DelaunayMeshing::ReadData() {
    {
        std::ifstream iFileT(input_path_ + "/triangulation_binary",std::ios::in | std::ios::binary);
        if (iFileT.fail()) throw std::runtime_error("failed to open output_triangulation_binary");
        CGAL::set_binary_mode(iFileT);
        iFileT >> trianglation_;

        std::cout << "trianglation_.number_of_vertices() " << trianglation_.number_of_vertices() << std::endl;
        std::cout << "trianglation_.number_of_cells() " << trianglation_.number_of_cells() << std::endl;
        std::cout << "trianglation_.number_of_edges() " << trianglation_.number_of_edges() << std::endl;
        std::cout << "trianglation_.number_of_facets() " << trianglation_.number_of_facets() << std::endl;
    }

    {
        std::ifstream in_file(input_label_file_name_);
        if (in_file.fail()) throw std::runtime_error("failed to open " + input_label_file_name_);
        std::istream_iterator<std::string> begin_iter(in_file);
        std::istream_iterator<std::string> end_iter;
        int element_count = std::distance(begin_iter, end_iter);
        in_file.close();
        in_file.open(input_label_file_name_);
        if (in_file.fail()) throw std::runtime_error("failed to open " + input_label_file_name_);

        cell_labels_.resize(element_count);
        for(int i=0; i<element_count; i++) {
            in_file >> cell_labels_[i];
        }
        in_file.close();
        std::cout << "cell label size() " << cell_labels_.size() << std::endl;
    }
}

void DelaunayMeshing::CreateMesh() {
    int cnt = 0;
    for (auto it = trianglation_.all_cells_begin(); it != trianglation_.all_cells_end(); ++it) {
        cell_label_map_[it] = cell_labels_[cnt];
        cnt++;
    }

    std::vector<Delaunay::Facet> surface_facets;
    std::unordered_set<Delaunay::Vertex_handle> surface_vertices;

    for(auto facet_it=trianglation_.finite_facets_begin(); facet_it!=trianglation_.finite_facets_end(); ++facet_it) {
        int cell_label = cell_label_map_.at(facet_it->first);
        int mirror_cell_label = cell_label_map_.at(facet_it->first->neighbor(facet_it->second));
        if(cell_label == mirror_cell_label) {
            continue;
        }

        for(int i=0; i<3; i++) {
            const auto& vertex = facet_it->first->vertex(trianglation_.vertex_triple_index(facet_it->second, i));
            surface_vertices.insert(vertex);
        }

        if(cell_label == 1) {
            surface_facets.push_back(*facet_it);
        } else {
            surface_facets.push_back(trianglation_.mirror_facet(*facet_it));
        }
    }

    open3d::geometry::TriangleMesh result_mesh;
    std::unordered_map<const Delaunay::Vertex_handle, int> surface_vertex_indices;
    surface_vertex_indices.reserve(surface_vertices.size());
    result_mesh.vertices_.reserve(surface_vertices.size());

    for(const auto& vertex : surface_vertices) {
        result_mesh.vertices_.push_back(CGALToEigen(vertex->point()));
        surface_vertex_indices.emplace(vertex, surface_vertex_indices.size());
    }

   result_mesh.triangles_.reserve(surface_facets.size());

   for(int i=0; i<surface_facets.size(); i++) {
       const auto& facet = surface_facets[i];
       result_mesh.triangles_.push_back(Eigen::Vector3i(surface_vertex_indices.at(facet.first->vertex(trianglation_.vertex_triple_index(facet.second, 0))),
                                         surface_vertex_indices.at(facet.first->vertex(trianglation_.vertex_triple_index(facet.second, 1))),
                                         surface_vertex_indices.at(facet.first->vertex(trianglation_.vertex_triple_index(facet.second, 2)))
                                         ));
   }

   open3d::io::WriteTriangleMesh(output_mesh_file_name_, result_mesh);
}

void DelaunayMeshing::Run() {
    ReadData();
    CreateMesh();
}

int main(int argc, char** argv) {
    std::string input_path = argv[1];
    std::string input_label_file_name = argv[2];
    std::string output_mesh_file_name = argv[3];

    DelaunayMeshing delaunay_meshing(input_path, input_label_file_name,output_mesh_file_name);
    delaunay_meshing.Run();
    std::cout << "Create mesh done!" << std::endl;
    return 0;
}

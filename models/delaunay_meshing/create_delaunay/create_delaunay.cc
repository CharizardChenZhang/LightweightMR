#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Kernel/global_functions.h>
#include <fstream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <iostream>
#include <unordered_map>
#include <random>
#include <open3d/Open3D.h>
using namespace std;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<int, K> Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K>                 Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb>                Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds, CGAL::Fast_location> Delaunay;

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

void ReadData(const std::string& data_path, open3d::geometry::PointCloud& original_points) {
//    if (!CGAL::IO::read_points(data_path, std::back_inserter(points))) {
//        std::cerr << "Error: cannot read file " << data_path << std::endl;
//        return EXIT_FAILURE;
//    }
    open3d::io::ReadPointCloud(data_path, original_points);
    std::cerr << "Read point data done!" << std::endl;
}

Delaunay ConstructDelaunay(const open3d::geometry::PointCloud& sampled_points) {
    std::vector<Delaunay::Point> delaunay_points(sampled_points.points_.size());
    for (size_t i = 0; i < sampled_points.points_.size(); ++i) {
        delaunay_points[i] = Delaunay::Point(sampled_points.points_[i][0], sampled_points.points_[i][1], sampled_points.points_[i][2]);
    }
    std::vector<int> point_idxs(sampled_points.points_.size());
    std::iota(point_idxs.begin(), point_idxs.end(), 0);

    std::cout << "Create triangulartion" << std::endl;
    const auto triangulation = Delaunay(boost::make_zip_iterator(boost::make_tuple(delaunay_points.begin(), point_idxs.begin())),
        boost::make_zip_iterator(boost::make_tuple(delaunay_points.end(), point_idxs.end())));
    return triangulation;
}

void compute_cell_vertex(Delaunay& t, std::vector<Eigen::Vector3d>& infinite_vertex, std::vector<int>& infinite_list) {
    int cnt = 0;
    for (auto ch = t.all_cells_begin(); ch != t.all_cells_end(); ch++) {
        std::vector<Eigen::Vector3d> vs;
        if (t.is_infinite(ch)) {
            for (int i = 0; i < 4; i++) {
                auto vh = ch->vertex(i);
                if (t.is_infinite(vh)) {
                    continue;
                }
                else {
                    vs.push_back(Eigen::Vector3d(vh->point().x(), vh->point().y(), vh->point().z()));
                }
            }
            for (int i = 0; i < 4; i++) {
                auto nei_ch = ch->neighbor(i);
                if (t.is_infinite(nei_ch)) {
                    continue;
                }
                else {
                    auto mirror_vh = t.mirror_vertex(ch, i);
                    vs.push_back(Eigen::Vector3d(mirror_vh->point().x(), mirror_vh->point().y(), mirror_vh->point().z()));
                    break;
                }
            }

            Eigen::Vector3d ray = (vs[0] + vs[1] + vs[2]) / 3.0 - vs[3];
            vs[3] = 100.0 * ray + (vs[0] + vs[1] + vs[2]) / 3.0;
            infinite_list.push_back(cnt);
            infinite_vertex.push_back(vs[3]);
        }
        cnt++;
    }
}

void write_delaunay(Delaunay triangulation, std::string output_path, std::vector<Eigen::Vector3d> infinite_vertex, std::vector<int> infinite_list) {
    {
        std::ofstream oFile_T(output_path + "/triangulation_binary", std::ios::out | std::ios::binary);
        CGAL::set_binary_mode(oFile_T);
        oFile_T << triangulation;
        oFile_T.close();
    }

    {
        std::ofstream oFile_infvertex(output_path + "/infinite_vertex.txt", std::ios::out);
        oFile_infvertex << std::fixed << std::setprecision(14);
        for (auto it = infinite_vertex.begin(); it != infinite_vertex.end(); ++it) {
            oFile_infvertex << (*it).transpose() << std::endl;
        }
        oFile_infvertex.close();
    }

    {
        std::ofstream oFile_infcell(output_path + "/infinite_cell_id.txt", std::ios::out);
        for (auto it = infinite_list.begin(); it != infinite_list.end(); ++it) {
            oFile_infcell << *it << " ";
            oFile_infcell << std::endl;
        }
        oFile_infcell.close();
    }

    {
        std::ofstream oFile_vertexid(output_path + "/cell_vertex_id.txt",std::ios::out);
        for (auto it = triangulation.all_cells_begin(); it != triangulation.all_cells_end(); ++it) {
            for (int j=0; j<4; j++) {
                if (triangulation.is_infinite(it->vertex(j))) {
                    oFile_vertexid << -1 << " ";
                } else {
                    oFile_vertexid << it->vertex(j)->info() << " ";
                }
                if (j == 3) {
                    oFile_vertexid << std::endl;
                }
            }
        }
        oFile_vertexid.close();
    }

    {
        CGAL::Unique_hash_map<Delaunay::Cell_handle, int> cell_idx_map;
        int cell_count = 0;
        for (auto it = triangulation.all_cells_begin(); it != triangulation.all_cells_end(); ++it) {
            cell_idx_map[it] = cell_count;
            cell_count++;
        }

        std::ofstream oFileT_adj(output_path + "/cell_adj_id.txt",std::ios::out );
        for (auto it = triangulation.all_cells_begin(); it != triangulation.all_cells_end(); ++it) {
            for(int i=0;i<4;i++) {
                auto neighbor_cell = it->neighbor(i);
                int neighbor_index = cell_idx_map[neighbor_cell];
                oFileT_adj << neighbor_index<<" ";
            }
            oFileT_adj << std::endl;
        }
        oFileT_adj.close();
    }
}

int main(int argc, char* argv[]){
    std::string input_point_path = argv[1];
    std::string output_path = argv[2];

    open3d::geometry::PointCloud original_points;
    ReadData(input_point_path, original_points);
    Delaunay triangulation = ConstructDelaunay(original_points);

    std::vector<Eigen::Vector3d> infinite_vertex;
    std::vector<int> infinite_list;
    compute_cell_vertex(triangulation, infinite_vertex, infinite_list);
    write_delaunay(triangulation, output_path, infinite_vertex, infinite_list);
    std::cout << "write_delaunay done" << std::endl;
}

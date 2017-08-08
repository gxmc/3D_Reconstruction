//
// Created by user on 8/6/17.
//
#include "simplify_mesh.h"
#include "openmvs.h"

// Constructor
OpenMVS::OpenMVS(fs::path const & dir) :
        reconstruction_dir(dir.parent_path())
{
    densify_path = local_path::OPENMVS_BIN / "DensifyPointCloud ";
    mesh_reconstruction_path = local_path::OPENMVS_BIN / "ReconstructMesh ";
    mesh_refinement_path = local_path::OPENMVS_BIN / "RefineMesh ";
    mesh_texture_path = local_path::OPENMVS_BIN / "TextureMesh ";
    interface_mvs_path = local_path::OPENMVS_BIN / "InterfaceVisualSFM ";
    scene = MVS::Scene(8);
}

// ----------- 0. Convert colmap NVM format to OpenMVS MVS format -----------
void OpenMVS::convert_from_nvm_to_mvs() {
    std::cout << "6. Convert model.nvm to scene.mvs" << std::endl;
    // Prepare args
    std::string working_dir_arg(" -w " + reconstruction_dir.string());
    std::string input_file_arg(" -i model.nvm");
    std::string output_dir_arg(" -o scene.mvs");
    // Run
    std::string converting(interface_mvs_path.string() + working_dir_arg + input_file_arg + output_dir_arg);
    success_on_previous_step = !system(converting.c_str());
}

// ----------- 1. Sparse point cloud densifying -----------
void OpenMVS::densify_point_cloud() {
    std::cout << "7. Densify point cloud" << std::endl;
    // Prepare args
    std::string working_path_arg(" -w " + reconstruction_dir.string());
    std::string input_path_arg(" -i scene.mvs");
    std::string output_path_arg(" -o scene_dense.mvs");
    std::string params(" --process-priority 1 --resolution-level 1");
    // Run
    std::string densifying(densify_path.string() + working_path_arg + input_path_arg + output_path_arg + params);
    success_on_previous_step = !system(densifying.c_str());
}

// ----------- 2. Remove NAN points after densifying -----------
void OpenMVS::remove_nan_values(MVS::PointCloud::PointArr & cloud) {
    std::cout << "Total points number in dense cloud: " << cloud.size() << std::endl;
    // Remove point from point cloud if it's any coordinate is NAN value
    TD_TIMER_START();
    for (auto it = cloud.begin(); it != cloud.end(); ++it) {
        if (std::isnan(it->x) || std::isnan(it->y) || std::isnan(it->z)) {
            cloud.erase(it);
        }
    }
    printf("Points number in dense cloud without NAN values: %lu (%s)\n", cloud.size(), TD_TIMER_GET_FMT().c_str());
}

void OpenMVS::remove_nan_points() {
    std::cout << "8. Removing NAN values from dense point cloud " << std::endl;
    // Load dense scene
    std::string input_file_arg(reconstruction_dir.string() + "/scene_dense.mvs");
    scene.Load(input_file_arg);
    if (scene.IsEmpty()) {
        success_on_previous_step = false;
        return;
    }
    // Removing NAN values from dense cloud, save it and scene
    remove_nan_values(scene.pointcloud.points);
    success_on_previous_step = !scene.pointcloud.points.IsEmpty(); // success if vector is NOT empty
    std::string path_to_output_scene = reconstruction_dir.string() + "/scene_dense.mvs";
    std::string path_to_output_cloud = reconstruction_dir.string() + "/scene_dense_without_nan.ply";
    scene.Save(path_to_output_scene);
    scene.pointcloud.Save(path_to_output_cloud);
    scene.Release();
}

// ----------- 3. Mesh reconstruction -----------
void OpenMVS::reconstruct_mesh(float const d = 7.0) {
    std::cout << "9. Reconstruct the mesh " << std::endl;
    // Prepare args
    std::string input_file_arg(" -i scene_dense.mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" -d " + std::to_string(d) + " --process-priority 1 --thickness-factor 1.0 --quality-factor 2.5 --close-holes 30 --smooth 3");
    // Run
    std::string reconstruction(mesh_reconstruction_path.string() + working_dir + input_file_arg + params);
    success_on_previous_step = !system(reconstruction.c_str());
}

// ----------- 4. Mesh refinement -----------
void OpenMVS::refining_mesh() {
    std::cout << "10. Refine the mesh " << std::endl;
    // Prepare args
    std::string input_file_arg(" -i scene_dense_mesh.mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" --process-priority 1 --resolution-level 0 --ensure-edge-size 2 --close-holes 30");
    // Run
    std::string refinement(mesh_refinement_path.string() + working_dir + input_file_arg + params);
    success_on_previous_step = !system(refinement.c_str());
}

// ----------- 5. Resize the mesh -----------

// Convert double to string with 2 sign after comma: 0.00
std::string OpenMVS::double_to_string(double val) {
    val = round(val * 10) / 10;
    return std::to_string(val).substr(0, 4);
}

// Push scene vertices and triangles to refined_mesh
template <class L, class R>
void fill_simplify_mesh(L const &from, std::vector<R> &to) {
    R tmp; // create Vertex or Triangle
    for (auto it = from.cbegin(); it != from.cend(); ++it) {
        tmp.update(it->x, it->y, it->z);
        to.push_back(tmp); // push vertex or triangle to vector for mesh refinement
    }
};

// Calculate target faces count for refined mesh
ulong calc_target_faces_count(std::vector<MeshSimplify::Vertex> const &simplified_mesh_vertices,
                              std::vector<MeshSimplify::Triangle> const &simplified_mesh_faces,
                              double const ratio)
{
    if ((simplified_mesh_faces.size() < 3) || (simplified_mesh_vertices.size() < 3)) {
        return 0;
    }
    ulong target_count = (ulong)round((double)simplified_mesh_faces.size() * ratio);

    if (target_count < 4) {
        printf("Object will not survive such extreme decimation\n");
        return 0;
    }
    printf("Input: %zu vertices, %zu triangles (target %lu)\n",
           simplified_mesh_vertices.size(), simplified_mesh_faces.size(), target_count);
    return target_count;
}

// Push vertices and triangles after mesh refinement to scene back
template <class L, class R, class Elem>
void fill_scene_mesh(std::vector<L> & from, R & to) {
    to.Reset();
    for (auto it = from.cbegin(); it != from.cend(); ++it) {
        float x, y, z;
        std::tie(x, y, z) = it->get_coord();
        to.push_back(Elem(x, y, z));
    }
};

// main function for mesh simplifying
void OpenMVS::simplify_mesh(double ratio = 0.5, double aggressiveness = 7.0) {
    printf("Mesh Simplification (C)2014 by Sven Forstmann in 2014, MIT License (%zu-bit)\n", sizeof(size_t) * 8);

    clock_t start = clock();
    // Load refined mesh
    scene.Load(reconstruction_dir.string() + "/scene_dense_mesh_refine.mvs");
    if (scene.IsEmpty()) {
        success_on_previous_step = false;
        return;
    }

    ulong v_count = scene.mesh.vertices.size();
    ulong f_count = scene.mesh.faces.size();
    MVS::Mesh::VertexArr & scene_vertices = scene.mesh.vertices;
    MVS::Mesh::FaceArr & scene_faces = scene.mesh.faces;

    MeshSimplify mesh(v_count, f_count);
    std::vector<MeshSimplify::Triangle> & simplified_mesh_faces = mesh.triangles;
    std::vector<MeshSimplify::Vertex> & simplified_mesh_vertices = mesh.vertices;

    // Push vertices and triangles from scene to temporary mesh for simplifying
    fill_simplify_mesh<MVS::Mesh::VertexArr, MeshSimplify::Vertex>(scene_vertices, simplified_mesh_vertices);
    fill_simplify_mesh<MVS::Mesh::FaceArr, MeshSimplify::Triangle>(scene_faces, simplified_mesh_faces);

    ulong target_count = calc_target_faces_count(simplified_mesh_vertices, simplified_mesh_faces, ratio);

    ulong startSize = simplified_mesh_faces.size();
    if (startSize == 0) {
        printf("Initial mesh is empty!\n");
        return;
    }

    // Reduce mesh faces(triangles) from initial to target count
    mesh.simplify_mesh(target_count, aggressiveness, true);

    if (simplified_mesh_faces.size() >= startSize) {
        printf("Unable to reduce mesh.\n");
    }
    // Push vertices and triangles to scene from temporary mesh after simplifying
    fill_scene_mesh<MeshSimplify::Vertex, MVS::Mesh::VertexArr, MVS::Mesh::Vertex>(simplified_mesh_vertices, scene_vertices);
    fill_scene_mesh<MeshSimplify::Triangle, MVS::Mesh::FaceArr, MVS::Mesh::Face>(simplified_mesh_faces, scene_faces);

    // Save simplified mesh to scene for texture and to ply format
    scene.Save(reconstruction_dir.string() + "/scene_dense_mesh_refine_resized.mvs");
    scene.mesh.Save(reconstruction_dir.string() + "/scene_dense_mesh_refine_resized.ply");
    printf("Output: %zu vertices, %zu triangles (%f reduction; %.4f sec)\n",
           simplified_mesh_vertices.size(), simplified_mesh_faces.size(),
           (float)simplified_mesh_faces.size() / (float) startSize , ((float)(clock() -start))  / CLOCKS_PER_SEC );
    scene.Release();
}

// ----------- 6. Texture the mesh -----------
fs::path OpenMVS::texture_mesh(double const ratio = 0.0) {
    std::cout << "12. Texture the remeshed model " << std::endl;
    std::string str_ratio = double_to_string(ratio);
    // Prepare args
    std::string input_file_arg(" -i scene_dense_mesh_refine_resized.mvs");
    std::string output_file_arg(" -o scene_texture_" + str_ratio + ".mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" --process-priority 1 ");
    // Run
    std::string texture(mesh_texture_path.string() + working_dir + input_file_arg + output_file_arg + params);
    success_on_previous_step = !system(texture.c_str());
    return reconstruction_dir / ("scene_texture_" + str_ratio + ".mvs");
}

// ----------- 7. Centering the mesh -----------
void OpenMVS::centering_textured_mesh(fs::path const & textured_mesh_path) {
    std::cout << "13. Centering the textured mesh " << std::endl;
    TD_TIMER_START();
    //  Load scene
    scene.Load(textured_mesh_path.string());
    if (scene.IsEmpty()) {
        success_on_previous_step = false;
        return;
    }
    // Centering textured mesh
    cv::Point3d centroid(0, 0, 0);
    for (auto it = scene.mesh.vertices.begin(); it != scene.mesh.vertices.end(); ++it) {
        centroid.x += it->x;
        centroid.y += it->y;
        centroid.z += it->z;
    }
    centroid = centroid / int(scene.mesh.vertices.size());
    for (auto it = scene.mesh.vertices.begin(); it != scene.mesh.vertices.end(); ++it) {
        it->x -= centroid.x;
        it->y -= centroid.y;
        it->z -= centroid.z;
    }
    // Save final mesh
    scene.mesh.Save(reconstruction_dir.string() + "/scene_centered.obj");
    printf("Textured mesh has centered: %s\n", TD_TIMER_GET_FMT().c_str());
    scene.Release();
}

// ----------- pipeline -----------
void OpenMVS::build_model_from_sparse_point_cloud() {
    if (success_on_previous_step) convert_from_nvm_to_mvs();
    if (success_on_previous_step) densify_point_cloud();
    if (success_on_previous_step) remove_nan_points();
    if (success_on_previous_step) reconstruct_mesh();
    if (success_on_previous_step) refining_mesh();
    if (success_on_previous_step) simplify_mesh();
    fs::path path;
    if (success_on_previous_step) path = texture_mesh();
    if (success_on_previous_step) centering_textured_mesh(path);
}
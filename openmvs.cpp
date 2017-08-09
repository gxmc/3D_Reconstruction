//
// Created by user on 8/6/17.
//
#include "simplify_mesh.h"
#include "openmvs.h"

// Convert double to string with 2 sign after comma: 0.00
std::string double_to_string(double val) {
    val = round(val * 10) / 10;
    return std::to_string(val).substr(0, 4);
}

// Constructor
OpenMVS::OpenMVS(fs::path const & dir, bool set_automatic_execution = true) :
        reconstruction_dir(dir.parent_path()), automatic_execution(set_automatic_execution)
{
    densify_path = local_path::OPENMVS_BIN / "DensifyPointCloud ";
    mesh_reconstruction_path = local_path::OPENMVS_BIN / "ReconstructMesh ";
    mesh_refinement_path = local_path::OPENMVS_BIN / "RefineMesh ";
    mesh_texture_path = local_path::OPENMVS_BIN / "TextureMesh ";
    interface_mvs_path = local_path::OPENMVS_BIN / "InterfaceVisualSFM ";
    scene = MVS::Scene(8);
}

bool OpenMVS::get_status() const {
    return success_on_previous_step;
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
void remove_nan_values(MVS::PointCloud::PointArr & cloud) {
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
    // Load dense scene
    std::string input_file_arg(reconstruction_dir.string() + "/scene_dense.mvs");
    scene.Load(input_file_arg);
    if (scene.IsEmpty() || scene.pointcloud.IsEmpty()) {
        success_on_previous_step = false;
        return;
    }
    std::cout << "8. Removing NAN values from dense point cloud " << std::endl;
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
void OpenMVS::reconstruct_mesh(double const dist = 7.0) {
    std::cout << "9. Reconstruct the mesh " << std::endl;
    std::string distance = double_to_string(dist);
    // Prepare args
    std::string input_file_arg(" -i scene_dense.mvs");
    std::string output_file_arg(" -o dense_mesh_" + distance + ".mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" -d " + distance + " --process-priority 1 --thickness-factor 1.0 --quality-factor 2.5 --close-holes 30 --smooth 3");
    // Run
    std::string reconstruction(mesh_reconstruction_path.string() + working_dir + input_file_arg + output_file_arg + params);
    success_on_previous_step = !system(reconstruction.c_str());
    common_distance_param = distance;
    common_simplify_ratio_param = "";
}

// ----------- 4. Mesh refinement -----------
void OpenMVS::refining_mesh() {
    std::cout << "10. Refine the mesh " << std::endl;
    // Prepare args
    std::string input_file_arg(" -i dense_mesh_" + common_distance_param + ".mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" --process-priority 1 --resolution-level 0 --ensure-edge-size 2 --close-holes 30");
    // Run
    std::string refinement(mesh_refinement_path.string() + working_dir + input_file_arg + params);
    success_on_previous_step = !system(refinement.c_str());
}

// ----------- 5. Resize the mesh -----------
// Push scene vertices and triangles to refined_mesh
template <class L, class R>
void fill_simplify_mesh(L const & from, std::vector<R> & to) {
    R tmp; // create Vertex or Triangle
    for (auto it = from.cbegin(); it != from.cend(); ++it) {
        tmp.update(it->x, it->y, it->z);
        to.push_back(tmp); // push vertex or triangle to vector for mesh refinement
    }
};

// Calculate target faces count for refined mesh
ulong calc_target_faces_count(std::vector<MeshSimplify::Vertex> const & simplified_mesh_vertices,
                              std::vector<MeshSimplify::Triangle> const & simplified_mesh_faces,
                              double ratio)
{
    if ((simplified_mesh_faces.size() < 3) || (simplified_mesh_vertices.size() < 3)) {
        return 0;
    }
    ulong target_count = (ulong)round((double)simplified_mesh_faces.size() * ratio);

    while (target_count < 2000) {
        std::cout << "New mesh will contain " << target_count << " faces. "
                "We can't allow this. It should contain at least 2000 faces.\n";
        std::cout << "Min param value: " << 2000.0 / (double)simplified_mesh_faces.size() << std::endl;
        std::cout << "Please input param from 0 to 1. For example 0.2 will decimate 80% of triangles:" << std::endl;
        std::cin >> ratio;
        target_count = calc_target_faces_count(simplified_mesh_vertices, simplified_mesh_faces, ratio);
    }
    printf("Input: %zu vertices, %zu triangles (target %lu)\n",
           simplified_mesh_vertices.size(), simplified_mesh_faces.size(), target_count);
    return target_count;
}

// Push vertices and triangles after mesh refinement to scene back
template <class L, class R, class Elem>
void fill_scene_mesh(std::vector<L> const & from, R & to) {
    to.Reset();
    for (auto it = from.cbegin(); it != from.cend(); ++it) {
        float x, y, z;
        std::tie(x, y, z) = it->get_coord();
        to.push_back(Elem(x, y, z));
    }
};

// main function for mesh simplifying
void OpenMVS::simplify_mesh(double ratio = 0.5, double const aggressiveness = 7.0) {
    clock_t start = clock();
    // Load refined mesh
    scene.Load(reconstruction_dir.string() + "/dense_mesh_" + common_distance_param + "_refine.mvs");
    if (scene.IsEmpty()) {
        success_on_previous_step = false;
        return;
    }
    printf("Mesh Simplification (C)2014 by Sven Forstmann in 2014, MIT License (%zu-bit)\n", sizeof(size_t) * 8);
    // Get vertices and faces from scene
    ulong v_count = scene.mesh.vertices.size();
    ulong f_count = scene.mesh.faces.size();
    MVS::Mesh::VertexArr & scene_vertices = scene.mesh.vertices;
    MVS::Mesh::FaceArr & scene_faces = scene.mesh.faces;

    // Push vertices and triangles from scene to temporary mesh for simplifying
    MeshSimplify mesh(v_count, f_count);
    std::vector<MeshSimplify::Triangle> & simplified_mesh_faces = mesh.triangles;
    std::vector<MeshSimplify::Vertex> & simplified_mesh_vertices = mesh.vertices;
    fill_simplify_mesh<MVS::Mesh::VertexArr, MeshSimplify::Vertex>(scene_vertices, simplified_mesh_vertices);
    fill_simplify_mesh<MVS::Mesh::FaceArr, MeshSimplify::Triangle>(scene_faces, simplified_mesh_faces);

    // Reduce mesh faces(triangles) from initial to target count
    ulong target_count = calc_target_faces_count(simplified_mesh_vertices, simplified_mesh_faces, ratio);
    mesh.simplify_mesh(target_count, aggressiveness, true);

    // Push vertices and triangles to scene from temporary mesh after simplifying
    fill_scene_mesh<MeshSimplify::Vertex, MVS::Mesh::VertexArr, MVS::Mesh::Vertex>(simplified_mesh_vertices, scene_vertices);
    fill_scene_mesh<MeshSimplify::Triangle, MVS::Mesh::FaceArr, MVS::Mesh::Face>(simplified_mesh_faces, scene_faces);

    // Save simplified mesh to scene for texture and to ply format
    std::string simplify_ratio = double_to_string(ratio);
    scene.Save(reconstruction_dir.string() +
                       "/dense_mesh_" + common_distance_param + "_refine_" + simplify_ratio + "_resized.mvs");
    scene.mesh.Save(reconstruction_dir.string() +
                            "/dense_mesh_" + common_distance_param + "_refine_" + simplify_ratio + "_resized.ply");
    printf("Output: %zu vertices, %zu triangles (%f reduction; %.4f sec)\n",
           simplified_mesh_vertices.size(), simplified_mesh_faces.size(),
           (float)simplified_mesh_faces.size() / (float) scene_faces.size(), ((float)(clock() -start))  / CLOCKS_PER_SEC);

    // Reset scene: clean vertices and faces. Remember simplify_ratio for texture step (input file has ration in filename)
    scene.Release();
    common_simplify_ratio_param = simplify_ratio;
    // Set bool simplified value as true
    simplified = true;
}

// ----------- 6. Texture the mesh -----------
fs::path OpenMVS::texture_mesh() {
    std::cout << "12. Texture the remeshed model " << std::endl;
    // Prepare args
    std::string input_file_arg;
    // Input file depends on parameter 'simplify_ratio' from simplify_mesh(...) {...}
    if (common_simplify_ratio_param == "") {
        input_file_arg = " -i dense_mesh_" + common_distance_param + "_refine.mvs";
    } else {
        input_file_arg = " -i dense_mesh_" + common_distance_param + "_refine_" +
                common_simplify_ratio_param + "_resized.mvs";
    }
    std::string output_file_arg(" -o texture_" + common_distance_param + "_" + common_simplify_ratio_param + ".mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" --process-priority 1 ");
    // Run
    std::string texture(mesh_texture_path.string() + working_dir + input_file_arg + output_file_arg + params);
    // If texture failed we can try again with another mesh
    // Success is determined by god. Do not simplify mesh at all or leave more faces in simplified mesh.
    if (system(texture.c_str())) {
        std::cerr << "Can't texture mesh. Increase it's face amount!" << std::endl;
        success_on_previous_step = true;
        return fs::path();
    }
    return reconstruction_dir / ("texture_" + common_distance_param + "_" + common_simplify_ratio_param + ".mvs");
}

// ----------- 7. Centering the mesh -----------
void OpenMVS::centering_textured_mesh(fs::path const & textured_mesh_path) {
    TD_TIMER_START();
    //  Load scene
    scene.Load(textured_mesh_path.string());
    if (scene.IsEmpty()) {
        return;
    }
    std::cout << "13. Centering the textured mesh " << std::endl;
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
    scene.mesh.Save(reconstruction_dir.parent_path().parent_path().string() +
                            "/texture_" + common_distance_param + "_" + common_simplify_ratio_param + "centered.obj");
    printf("Textured mesh has centered: %s\n", TD_TIMER_GET_FMT().c_str());
    scene.Release();
}

// Command line interface. Working for two steps: Mesh Simplifying and Mesh Reconstruction
// flag - bool value for "to do"/ "not to do" simplify.
// ratio - double value for:
// 1) In Mesh Simplifying case: simplify_ratio
// 2) In Mesh Reconstruction case: distance
int dialog(std::string const & question, std::string const & suggestion, bool & simplify,
           double & value, double const min, double const max)
{
    std::cout << question << std::endl;
    std::string answer;
    // Check desire of simplifying or reconstruction mesh
    while ((answer != "yes") && (answer != "no")) {
        std::cout << "Type 'yes' or 'no': " << std::endl;
        std::cin >> answer;
    }
    if (answer == "yes") {
        simplify = true;
        std::string x;
        while (1) {
            // Read double
            std::cout << suggestion << std::endl;
            std::cin >> x;
            value = atof(x.c_str());
            // Check constrains
            if ((min < value) && (value < max)) {
                break;
            }
        }
        return 1;
    } else {
        value = 0.0;
        return 0;
    }
}


// ----------- pipeline -----------
void OpenMVS::build_model_from_sparse_point_cloud() {
    if (success_on_previous_step) convert_from_nvm_to_mvs(); else return;
    if (success_on_previous_step) densify_point_cloud(); else return;
    if (success_on_previous_step) remove_nan_points();  else return;
    double distance = 7.0;
    double simplify_ratio = 0.0;
    fs::path path;
    // Try to build mesh from dense point cloud.
    while (true) {
        // Mesh is built with parameter 'distance'
        if (success_on_previous_step) reconstruct_mesh(distance); else return;
        // Then it is refined
        if (success_on_previous_step) refining_mesh(); else return;
        bool start_simplify = false;
        while (true) {
            // Mesh can be simplified. Default it is not simplified.
            if (start_simplify) {
                if (success_on_previous_step) simplify_mesh(simplify_ratio);
            }
            // Mesh texture and centering
            if (success_on_previous_step && simplified) path = texture_mesh();
            if (success_on_previous_step && simplified) centering_textured_mesh(path);

            if (!automatic_execution) {
                // Offering to user simplify mesh and retexture it with some parameter
                std::string question("Would you like to simplify mesh with another param: (yes, no)");
                std::string suggestion(
                        "Please input param from 0.1 to 1. For example 0.2 will decimate 80% of triangles:");
                if (!dialog(question, suggestion, start_simplify, simplify_ratio, 0.1, 1))
                    break;
            } else {
                break;
            }
        }
        if (!automatic_execution) {
            // Offering to user completely reconstruct mesh with some parameter
            std::string question("Would you like to reconstruct mesh with another param: (yes, no)");
            std::string suggestion("Now param value is 7.0. Please input param from 0.1 to 15:");
            if (!dialog(question, suggestion, start_simplify, distance, 0.1, 15))
                break;
        } else {
            break;
        }
    }
}
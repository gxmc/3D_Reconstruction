//
// Created by user on 8/6/17.
//

#include "Simplify.h"
#include "openmvs.h"

OpenMVS::OpenMVS(fs::path const & dir) :
        reconstruction_dir(dir.parent_path())
{
    densify_path = local_path::OPENMVS_BIN / "DensifyPointCloud ";
    mesh_reconstruction_path = local_path::OPENMVS_BIN / "ReconstructMesh ";
    mesh_refinement_path = local_path::OPENMVS_BIN / "RefineMesh ";
    mesh_texture_path = local_path::OPENMVS_BIN / "TextureMesh ";
    interface_mvs_path = local_path::OPENMVS_BIN / "InterfaceVisualSFM ";
}

void OpenMVS::convert_from_nvm_to_mvs() {
    std::cout << "6. Convert model.nvm to scene.mvs" << std::endl;

    // Prepare args
    std::string input_file_arg(" -i " + reconstruction_dir.string() + "/model.nvm");
    std::string working_dir_arg(" -w " + reconstruction_dir.string());
    std::string output_dir_arg(" -o " + reconstruction_dir.string() + "/scene.mvs");

    // Run
    std::string converting(interface_mvs_path.string() + input_file_arg + working_dir_arg + output_dir_arg);
    int converting_has_done = system(converting.c_str());
    assert(!converting_has_done);

}

void OpenMVS::densify_point_cloud() {
    std::cout << "7. Densify point cloud" << std::endl;

    // Prepare args
    std::string input_path_arg(" -i " + reconstruction_dir.string() + "/scene.mvs");
    std::string working_path_arg(" -w " + reconstruction_dir.string());
    std::string output_path_arg(" -o " + reconstruction_dir.string() + "/scene_dense.mvs");
    std::string params(" --process-priority 1 --resolution-level 1");

    // Run
    std::string densifying(densify_path.string() + input_path_arg + working_path_arg + output_path_arg + params);
    int densifying_has_done = system(densifying.c_str());
    assert(!densifying_has_done);
}

void OpenMVS::remove_nan_values(MVS::PointCloud::PointArr & cloud) {
    size_t points_from_scene_number = cloud.size();
    std::cout << "Total points number in dense cloud: " << points_from_scene_number << std::endl;
    TD_TIMER_START();
    for (unsigned long i = 0; i < points_from_scene_number; ++i) {
        cv::Point3d p = cloud[i];
        if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) {
            cloud.RemoveAt(i);
            --points_from_scene_number;
        }
    }
    printf("Points number in dense cloud without NAN values: %u (%s)\n", cloud.size(), TD_TIMER_GET_FMT().c_str());
}

void OpenMVS::remove_nan_points() {
    std::cout << "8. Removing NAN values from dense point cloud " << std::endl;

    // Load scene
    std::string input_file_arg(reconstruction_dir.string() + "/scene_dense.mvs");
    MVS::Scene scene(1);
    bool success = scene.Load(input_file_arg);
    if (!success) {
        std::cerr << "Can't open scene " + input_file_arg + " or scene doesn't exists.\nSet full path to scene: path/*.mvs" << std::endl;
    }

    // Removing NAN values and save scene
    remove_nan_values(scene.pointcloud.points);
    std::string path_to_output_scene = reconstruction_dir.string() + "/scene_dense.mvs";
    std::string path_to_output_cloud = reconstruction_dir.string() + "/scene_dense_without_nan.ply";
    scene.Save(path_to_output_scene);
    scene.pointcloud.Save(path_to_output_cloud);
}

void OpenMVS::reconstruct_mesh(float const d = 7.0) {
    std::cout << "9. Reconstruct the mesh " << std::endl;

    // Prepare args
    std::string input_file_arg(" -i " + reconstruction_dir.string() + "/scene_dense.mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" -d " + std::to_string(d) + " --process-priority 1 --thickness-factor 1.0 --quality-factor 2.5 --close-holes 30 --smooth 3");
    // Run
    std::string reconstruction(mesh_reconstruction_path.string() + input_file_arg + working_dir + params);
    int reconstruction_has_done = system(reconstruction.c_str());
    assert(!reconstruction_has_done);
}

void OpenMVS::refining_mesh() {
    std::cout << "10. Refine the mesh " << std::endl;

    // Prepare args
    std::string input_file_arg(" -i " + reconstruction_dir.string() + "/scene_dense_mesh.mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" --process-priority 1 --resolution-level 0 --ensure-edge-size 2 --close-holes 30");

    // Run
    std::string reconstruction(mesh_refinement_path.string() + input_file_arg + working_dir + params);
    int reconstruction_has_done = system(reconstruction.c_str());
    assert(!reconstruction_has_done);
}

std::string OpenMVS::double_to_string(double val) {
    val = round(val * 10) / 10;
    return std::to_string(val).substr(0, 4);
}

//template <class L, class R>
//void vector_assing(L & from, std::vector<R> & to) {
//    R tmp;
//    for (auto it = from.cbegin(); it != from.cend(); ++it) {
//        tmp = {*it[0], *it[1], *it[2]};
//        to.push_back(tmp);
//    }
//};

void OpenMVS::resize_mesh(double ratio = 0.5, double agressiveness = 7.0) {
    printf("Mesh Simplification (C)2014 by Sven Forstmann in 2014, MIT License (%zu-bit)\n", sizeof(size_t)*8);
    std::cout << "Resize mesh? (yes, no)" << std::endl;
    std::string answer;
//    std::cin >> answer;
//    if (answer == "no") {
//        return;
//    }
    clock_t start = clock();
    MVS::Scene scene(1);
    bool success = scene.Load(reconstruction_dir.string() + "/scene_dense_mesh_refine.mvs");
    assert(success);

    Simplify::vertices.reserve(scene.mesh.vertices.size());
    Simplify::Vertex v;

    for (auto i = 0; i < scene.mesh.vertices.size(); ++i) {
        v.p.x = scene.mesh.vertices[i].x;
        v.p.y = scene.mesh.vertices[i].y;
        v.p.z = scene.mesh.vertices[i].z;
        Simplify::vertices.push_back(v);
    }


    Simplify::triangles.reserve(scene.mesh.faces.size());
    Simplify::Triangle t;
    for (auto i = 0; i < scene.mesh.faces.size(); ++i) {
        t.v[0] = scene.mesh.faces[i].x;
        t.v[1] = scene.mesh.faces[i].y;
        t.v[2] = scene.mesh.faces[i].z;
        Simplify::triangles.push_back(t);
    }


    if ((Simplify::triangles.size() < 3) || (Simplify::vertices.size() < 3))
        return;
    int target_count = (int)round((double)Simplify::triangles.size() * ratio);

    if (target_count < 4) {
        printf("Object will not survive such extreme decimation\n");
        return;
    }

    printf("Input: %zu vertices, %zu triangles (target %d)\n", Simplify::vertices.size(), Simplify::triangles.size(), target_count);
    int startSize = Simplify::triangles.size();
    Simplify::simplify_mesh(target_count, agressiveness, true);

    if ( Simplify::triangles.size() >= startSize) {
        printf("Unable to reduce mesh.\n");
    }

    scene.mesh.vertices.Reset();
    MVS::Mesh::Vertex v_mvs;
    for (auto i = 0; i <  Simplify::vertices.size(); ++i) {
        v_mvs.x = Simplify::vertices[i].p.x;
        v_mvs.y = Simplify::vertices[i].p.y;
        v_mvs.z = Simplify::vertices[i].p.z;
        scene.mesh.vertices.push_back(v_mvs);
    }

    scene.mesh.faces.Reset();
    MVS::Mesh::Face f;
    for (auto i = 0; i < Simplify::triangles.size(); ++i) {
        f[0] = Simplify::triangles[i].v[0];
        f[1] = Simplify::triangles[i].v[1];
        f[2] = Simplify::triangles[i].v[2];
        scene.mesh.faces.push_back(f);
    }

    scene.Save(reconstruction_dir.string() + "/scene_dense_mesh_refine_resized.mvs");
    scene.mesh.Save(reconstruction_dir.string() + "/scene_dense_mesh_refine_resized.ply");
    printf("Output: %zu vertices, %zu triangles (%f reduction; %.4f sec)\n",Simplify::vertices.size(), Simplify::triangles.size()
            , (float)Simplify::triangles.size()/ (float) startSize  , ((float)(clock()-start))/CLOCKS_PER_SEC );
    Simplify::vertices.clear();
    Simplify::triangles.clear();
}

fs::path OpenMVS::texturing_mesh(double const ratio = 0.0) {
    std::cout << "12. Texture the remeshed model " << std::endl;
    std::string str_ratio = double_to_string(ratio);
//    std::string mesh_filename = "scene_dense_mesh_refine_resized" + str_ratio + ".ply";

    // Prepare args
    std::string input_file_arg(" -i scene_dense_mesh_refine_resized.mvs");
    std::string output_file_arg(" -o scene_texture_" + str_ratio + ".mvs");
    std::string working_dir(" -w " + reconstruction_dir.string());
    std::string params(" --process-priority 1 ");

    // Run
    std::string texturing(mesh_texture_path.string() + input_file_arg + output_file_arg + working_dir + params);
    int reconstruction_has_done = system(texturing.c_str());
    assert(!reconstruction_has_done);
    return reconstruction_dir / ("scene_texture_" + str_ratio + ".mvs");
}

void OpenMVS::centering_textured_mesh(fs::path const & textured_mesh_path) {
    std::cout << "13. Centering the textured mesh " << std::endl;
    TD_TIMER_START();
    //  Load scene
    MVS::Scene scene(8);
    bool success = scene.Load(textured_mesh_path.string());
    if (!success) {
        std::cerr << "Can't open scene " + textured_mesh_path.string() << std::endl;
    }

    // Removing NAN values and save scene
    cv::Point3d centroid(0, 0, 0);
    for (int i = 0; i < scene.mesh.vertices.size(); ++i) {
        centroid.x += scene.mesh.vertices[i].x;
        centroid.y += scene.mesh.vertices[i].y;
        centroid.z += scene.mesh.vertices[i].z;
    }
    centroid = centroid / int(scene.mesh.vertices.size());
    for (int i = 0; i < scene.mesh.vertices.size(); ++i) {
        scene.mesh.vertices[i].x -= centroid.x;
        scene.mesh.vertices[i].y -= centroid.y;
        scene.mesh.vertices[i].z -= centroid.z;
    }
    scene.mesh.Save(reconstruction_dir.string() + "/scene_centered.obj");
    printf("Textured mesh has centered: %s\n", TD_TIMER_GET_FMT().c_str());
}

void OpenMVS::build_model_from_sparse_point_cloud() {
    convert_from_nvm_to_mvs();
    densify_point_cloud();
    remove_nan_points();
    reconstruct_mesh();
    refining_mesh();
    resize_mesh();
    fs::path path = texturing_mesh();
    centering_textured_mesh(path);
}
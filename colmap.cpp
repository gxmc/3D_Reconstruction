//
// Created by user on 8/6/17.
//

#include "colmap.h"


void Colmap::extract_features() {
    std::cout << "1. Extract features" << std::endl;

    // Prepare args for feature_extractor_path
    std::string database_arg(" --database_path " + common_database.string());
    std::string image_path_arg(" --image_path " + input_dir.string());
    std::string single_camera_arg(" --ImageReader.single_camera 1");
    std::string use_gpu_arg(" --use_gpu 1");

    // Run colmap feature_extractor_path
    std::string feature_extractor(feature_extractor_path.string() + image_path_arg + database_arg + single_camera_arg + use_gpu_arg);
    std::cout << feature_extractor << std::endl;
    int feature_extractor_has_done = system(feature_extractor.c_str());
    assert(!feature_extractor_has_done);

    // Copy features database to sequential and exhaustive dirs
    fs::copy(common_database, sequential_dir / local_path::DATABASE_PATH, fs::copy_options::overwrite_existing);
    fs::copy(common_database, exhaustive_dir / local_path::DATABASE_PATH, fs::copy_options::overwrite_existing);
}

void Colmap::feature_matching(bool sequential) {
    std::cout << "2. Matching" << std::endl;
    fs::path current_database;
    std::string matcher;
    if (sequential) {
        current_database = sequential_dir / local_path::DATABASE_PATH;
        matcher = sequential_matcher_path.string();
    } else {
        current_database = exhaustive_dir / local_path::DATABASE_PATH;
        matcher = exhaustive_matcher_path.string();
    }

    // Prepare args for sequential_matcher_path
    std::string database_arg(" --database_path " + current_database.string());
    std::string num_threads(" --SiftMatching.num_threads 8");

    // Run colmap sequential_matcher_path
    matcher += (database_arg + num_threads);
    int feature_matcher_has_done = system(matcher.c_str());
    assert(!feature_matcher_has_done);
}

void Colmap::sparse_reconstruction(fs::path const & working_dir) {
    std::cout << "3. Sparse reconstruction" << std::endl;
    fs::path current_database;
    current_database = working_dir / local_path::DATABASE_PATH;

    fs::path export_path = current_database.parent_path() / "sparse";
    fs::create_directory(export_path);

    // Prepare args for sparse reconstruction
    std::string image_path_arg(" --image_path " + input_dir.string());
    std::string database_arg(" --database_path " + current_database.string());
    std::string export_path_arg(" --export_path " + export_path.string());
    std::string num_threads(" --Mapper.num_threads 8");

    // Run colmap sequential_matcher_path
    std::string sparse_reconstructor(mapper.string() + image_path_arg + database_arg + export_path_arg + num_threads);
    int sparse_reconstructor_has_done = system(sparse_reconstructor.c_str());
    assert(!sparse_reconstructor_has_done);
}

void Colmap::image_undistorting(fs::path const & working_dir) {
    std::cout << "4. Image undistorter" << std::endl;

    // Prepare args for sparse reconstruction
    std::string image_path_arg(" --image_path " + input_dir.string());
    std::string input_path_arg(" --input_path " + working_dir.string() + "/sparse/0");
    fs::create_directory(working_dir / "dense");
    std::string output_path_arg(" --output_path " + working_dir.string() + "/dense");
    std::string output_type_arg(" --output_type COLMAP");

    // Run colmap sequential_matcher_path
    std::string undistorting(image_undistorter_path.string() + image_path_arg + input_path_arg + output_path_arg + output_type_arg);
    int undistorting_has_done = system(undistorting.c_str());
    assert(!undistorting_has_done);
}

fs::path const Colmap::model_converting(fs::path const & working_dir) {
    std::cout << "5. Model converter" << std::endl;

    // Prepare args for sparse reconstruction
    std::string input_path_arg(" --input_path " + working_dir.string() + "/sparse/0");
    std::string nvm_model_path(" --output_path " + working_dir.string() + "/dense/images/model.nvm");
    std::string output_type_arg(" --output_type nvm");


    // Run colmap sequential_matcher_path
    std::string converting(model_converter_path.string() + input_path_arg + nvm_model_path + output_type_arg);
    int converting_has_done = system(converting.c_str());
    assert(!converting_has_done);
    return working_dir / "dense/images/model.nvm";
}

Colmap::Colmap(std::string const & image_dir, std::string const & colmap_bin_dir) :
    input_dir(image_dir),
    common_database(input_dir / local_path::DATABASE_PATH),
    sequential_dir(input_dir.parent_path() / local_path::SEQUENTIAL_PATH),
    exhaustive_dir(input_dir.parent_path() / local_path::EXHAUSTIVE_PATH)
{
    fs::path colmap_bin(colmap_bin_dir);
    feature_extractor_path = colmap_bin / "feature_extractor";
    sequential_matcher_path = colmap_bin / "sequential_matcher";
    exhaustive_matcher_path = colmap_bin / "exhaustive_matcher";
    mapper = colmap_bin / "mapper";
    image_undistorter_path = colmap_bin / "image_undistorter";
    model_converter_path = colmap_bin / "model_converter";
    extract_features();
}

fs::path const Colmap::sfm(bool sequential) {
    feature_matching(sequential);
    fs::path working_dir;
    if (sequential) {
        working_dir = sequential_dir;
    } else {
        working_dir = exhaustive_dir;
    }
    sparse_reconstruction(working_dir);
    image_undistorting(working_dir);
    return model_converting(working_dir);
}

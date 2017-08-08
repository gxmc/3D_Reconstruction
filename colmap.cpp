//
// Created by user on 8/6/17.
//

#include "colmap.h"

// Constructor
Colmap::Colmap(std::string const & image_dir, std::string const & colmap_bin_dir) :
        input_dir(image_dir),
        database(input_dir / local_path::DATABASE_PATH),
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
}

// COLMAP SFM pipeline (https://colmap.github.io/tutorial.html#structure-from-motion)
// 1. Feature extraction.
// 2. Matching (sequential or exhaustive)
// 3. Sparse reconstruction (camera positions, sparse point cloud, 2D-3D projections)
// 4. Image undistortion for correct dense reconstruction
// 5. Convert COLMAP data to NVM format. Then convert NVM to MVS format.

// ----------- 1. Perform feature extraction for a set of images. -----------
void Colmap::extract_features() {
    std::cout << "1. Extract features" << std::endl;

    // Prepare args for feature extractor
    std::string database_arg(" --database_path " + database.string());
    std::string image_path_arg(" --image_path " + input_dir.string());
    std::string single_camera_arg(" --ImageReader.single_camera 1");
    std::string use_gpu_arg(" --use_gpu 1");

    // Run colmap feature extractor
    std::string feature_extractor(feature_extractor_path.string() +
                                          image_path_arg + database_arg + single_camera_arg + use_gpu_arg);
    std::cout << "Run: " << feature_extractor << std::endl;
    success_on_previous_step = !system(feature_extractor.c_str());

    // Copy features database to sequential and exhaustive dirs
    fs::copy(database, sequential_dir / local_path::DATABASE_PATH, fs::copy_options::overwrite_existing);
    fs::copy(database, exhaustive_dir / local_path::DATABASE_PATH, fs::copy_options::overwrite_existing);
}

// ----------- 2. Perform feature matching after performing feature extraction. -----------
void Colmap::feature_matching(bool const sequential) {
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

    // Prepare args for sequential matcher
    std::string database_arg(" --database_path " + current_database.string());
    std::string num_threads(" --SiftMatching.num_threads 8");

    // Run colmap sequential matcher
    matcher += (database_arg + num_threads);
    std::cout << "Run: " << matcher << std::endl;
    success_on_previous_step = !system(matcher.c_str());
}

// ----------- 3. Sparse 3D reconstruction / mapping of the dataset using SfM
//                             after performing feature extraction and matching. -----------
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

    // Run colmap sparse reconstruction
    std::string sparse_reconstructor(mapper.string() + image_path_arg + database_arg + export_path_arg + num_threads);
    std::cout << "Run: " << sparse_reconstructor << std::endl;
    success_on_previous_step = !system(sparse_reconstructor.c_str()) && !fs::is_empty(export_path);
}

// ----------- 4. Remove the distortion from images -----------
void Colmap::image_undistorting(fs::path const & working_dir) {
    std::cout << "4. Image undistorter" << std::endl;

    // Prepare args for image undistorting
    std::string image_path_arg(" --image_path " + input_dir.string());
    std::string input_path_arg(" --input_path " + working_dir.string() + "/sparse/0");
    fs::create_directory(working_dir / "dense");
    std::string output_path_arg(" --output_path " + working_dir.string() + "/dense");
    std::string output_type_arg(" --output_type COLMAP");

    // Run colmap image undistorting
    std::string undistorting(image_undistorter_path.string() +
                                     image_path_arg + input_path_arg + output_path_arg + output_type_arg);
    std::cout << "Run: " << undistorting << std::endl;
    success_on_previous_step = !system(undistorting.c_str());
}

// ----------- 5. Convert COLMAP model to OpenVMS format -----------
fs::path Colmap::model_converting(fs::path const & working_dir) {
    std::cout << "5. Model converter" << std::endl;

    // Prepare args model converting from NVM to MVS
    std::string input_path_arg(" --input_path " + working_dir.string() + "/sparse/0");
    std::string nvm_model_path(" --output_path " + working_dir.string() + "/dense/images/model.nvm");
    std::string output_type_arg(" --output_type nvm");

    // Run colmap model converting
    std::string converting(model_converter_path.string() + input_path_arg + nvm_model_path + output_type_arg);
    std::cout << "Run: " << converting << std::endl;
    success_on_previous_step = !system(converting.c_str());
    return working_dir / "dense/images/model.nvm";
}


// ----------- Structure from Motion pipeline -----------
fs::path Colmap::sfm(bool sequential) {
    if (success_on_previous_step) extract_features();
    if (success_on_previous_step) feature_matching(sequential);
    fs::path working_dir;
    if (sequential) {
        working_dir = sequential_dir;
    } else {
        working_dir = exhaustive_dir;
    }
    if (success_on_previous_step) sparse_reconstruction(working_dir);
    if (success_on_previous_step) image_undistorting(working_dir);
    if (success_on_previous_step) {
        return model_converting(working_dir);
    } else {
        return fs::path();
    }
}

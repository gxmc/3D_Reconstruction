//
// Created by user on 8/6/17.
//

#ifndef RECONSTRUCTION_COLMAP_H
#define RECONSTRUCTION_COLMAP_H

#include "utils.h"

//# COLMAP SFM pipeline (https://colmap.github.io/tutorial.html#structure-from-motion)
//# 1. Feature extraction.
//# 2. Matching (sequential or exhaustive)
//# 3. Sparse reconstruction (camera positions, sparse point cloud, 2D-3D projections)
//# 4. Image undistortion for correct dense reconstruction
//# 5. Convert COLMAP data to NVM format. Then convert NVM to MVS format.

class Colmap {
    fs::path input_dir;
    fs::path common_database;
    fs::path sequential_dir;
    fs::path exhaustive_dir;
    fs::path feature_extractor_path;
    fs::path sequential_matcher_path;
    fs::path exhaustive_matcher_path;
    fs::path mapper;
    fs::path image_undistorter_path;
    fs::path model_converter_path;

    void extract_features();

    void feature_matching(bool sequential);

    void sparse_reconstruction(fs::path const & working_dir);

    void image_undistorting(fs::path const & working_dir);

    fs::path const model_converting(fs::path const & working_dir);

public:
    explicit Colmap(std::string const & image_dir, std::string const & colmap_bin_dir);

    fs::path const sfm(bool sequential);
};

#endif //RECONSTRUCTION_COLMAP_H

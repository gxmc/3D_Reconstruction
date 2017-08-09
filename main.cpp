#include <iostream>
#include <vector>

// Compiling from sources:
// 1) OpenMVS (https://github.com/cdcseacave/openMVS/wiki/Building)
// 2) COLMAP (https://colmap.github.io/install.html#build-from-source)

// For OpenMVS and others using:
// 1) Eigen 3.2.10 (3.3._ doesn't works)
// 2) Ceres-solver (http://ceres-solver.org/installation.html)

#include "image_processing.h"
#include "colmap.h"
#include "openmvs.h"

void reconstruction_pipeline(std::string const & working_dir, bool is_sequential, bool automatic = true) {
    TD_TIMER_START();
    // Run sequential SfM
    Colmap colmap(working_dir, local_path::COLMAP_BIN);
    fs::path const path_to_nvm_model = colmap.sfm(is_sequential);
    if (path_to_nvm_model.empty()) {
        std::cerr << "Reconstruction field!" << std::endl;
        return;
    }
    OpenMVS mvs(path_to_nvm_model, automatic);
    mvs.build_model_from_sparse_point_cloud();
    if (!mvs.get_status()) {
        std::cerr << "Reconstruction field!" << std::endl;
        return;
    }
    printf("Reconstruction consumed: %s\n", TD_TIMER_GET_FMT().c_str());
}


int main(int args, char* argv[]) {
    if (args < 2) {
        std::cout << "Using example:\n "
                "$./Reconstruction full_path_to_images(reqiued) "
                "automatic (1 or 0. If 0 you will choose params for mesh simplifying) "
                "full_path_colmap(optional, default '/usr/local/bin') "
                "full_path_openmvs(optional, default '/usr/local/bin/OpenMVS')" << std::endl;
        return 0;
    }
    fs::path input_dir = std::string(argv[1]);
    // This flag free from openmvs-dialog (see build_model_from_sparse_point_cloud(...) in openmvs.cpp)
    bool flag_automatic_execution = (bool)atoi(argv[2]);

    if (argv[3]) {
        local_path::COLMAP_BIN = fs::path(argv[3]);
    }
    if (argv[4]) {
        local_path::OPENMVS_BIN = fs::path(argv[4]);
    }

    // Image processing
    ImageProcessing processing(input_dir);
    processing.start();
    std::string working_dir = processing.get_working_dir();

    reconstruction_pipeline(working_dir, 1, flag_automatic_execution);
    reconstruction_pipeline(working_dir, 0, flag_automatic_execution);
    return 0;
}
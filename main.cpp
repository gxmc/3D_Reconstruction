#include <iostream>
#include <vector>

// Compiling from sources:
// 1) OpenMVS (https://github.com/cdcseacave/openMVS/wiki/Building)
// 2) COLMAP (https://colmap.github.io/install.html#build-from-source)
// 3) PyMesh (https://github.com/qnzhou/PyMesh#build)

// For OpenMVS and others using:
// 1) Eigen 3.2.10 (3.3._ doesn't works)
// 2) Ceres-solver (http://ceres-solver.org/installation.html)

#include <image_processing.h>
#include <colmap.h>
#include <openmvs.h>

void reconstruction_pipeline(std::string const & working_dir, bool is_sequential) {
    TD_TIMER_START();
    // Run sequential SfM
    Colmap colmap(working_dir, local_path::COLMAP_BIN);
    fs::path const path_to_nvm_model = colmap.sfm(is_sequential);

    OpenMVS mvs(path_to_nvm_model);
    mvs.build_model_from_sparse_point_cloud();
    printf("Reconstruction consumed: %s\n", TD_TIMER_GET_FMT().c_str());
}


int main(int args, char* argv[]) {
    std::cout << argv[1] << std::endl;
    fs::path input_dir = std::string(argv[1]);
//    std::cout << input_dir << std::endl;

    // Image processing
    ImageProcessing processing(input_dir);
    processing.start();
    std::string working_dir = processing.get_working_dir();

    reconstruction_pipeline(working_dir, 1);
    reconstruction_pipeline(working_dir, 0);
    return 0;
}
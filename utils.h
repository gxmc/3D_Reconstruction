//
// Created by user on 8/6/17.
//

#ifndef RECONSTRUCTION_UTILS_H
#define RECONSTRUCTION_UTILS_H

#include <iostream>
#include <experimental/filesystem>
#include <cassert>
namespace fs = std::experimental::filesystem::v1;

namespace local_path {
    static fs::path WORKING_PATH = "/result";
    static fs::path SEQUENTIAL_PATH = "/sequential_matching";
    static fs::path EXHAUSTIVE_PATH = "/exhaustive_matching";
    static fs::path DATABASE_PATH = "/database.db";
    static fs::path OPENMVS_BIN = "/usr/local/bin/OpenMVS";
    static fs::path COLMAP_BIN = "/usr/local/bin";
}

#endif //RECONSTRUCTION_UTILS_H

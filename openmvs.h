//
// Created by user on 8/6/17.
//

#ifndef RECONSTRUCTION_OPENMVS_H
#define RECONSTRUCTION_OPENMVS_H

#include <OpenMVS/MVS.h>
#include "utils.h"

// OpenMVS pipeline (https://github.com/cdcseacave/openMVS/wiki/Usage)
// It has 4 modules (https://github.com/cdcseacave/openMVS/wiki/Modules)
// 1. Sparse pointcloud densifying (http://www.connellybarnes.com/work/publications/2011_patchmatch_cacm.pdf)
//           (PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing C. Barnes et al. 2009).
//
// 2. Remove NAN points after densifying. (https://github.com/cdcseacave/openMVS/wiki/Interface)
//           (Using MVS interface)
//
// 3. Mesh reconstruction (https://www.hindawi.com/journals/isrn/2014/798595/)
//           (Exploiting Visibility Information in Surface Reconstruction
//               to Preserve Weakly Supported Surfaces M. Jancosek et al. 2014).
//
//   M. Jancosek dissertation: Large Scale Surface Reconstruction based on Point Visibility
//       (https://dspace.cvut.cz/bitstream/handle/10467/60872/Disertace_Jancosek_2014.pdf?sequence=1)
//
// 4. Mesh refinement (http://sci-hub.cc/http://ieeexplore.ieee.org/document/5989831/)
//           (High Accuracy and Visibility-Consistent Dense Multiview Stereo HH. Vu et al. 2012).
//
// 5. Mesh centering. (https://github.com/cdcseacave/openMVS/wiki/Interface)
//           (Using MVS interface)
//
// 6. Mesh texturing (https://pdfs.semanticscholar.org/c8e6/eefd01b17489d38c355cf21dd492cbd02dab.pdf)
//           (Let There Be Color! - Large-Scale Texturing of 3D Reconstructions M. Waechter et al. 2014).
//
// 7. Remeshing and retexturing (Pymesh + Mesh texturing) (http://pymesh.readthedocs.io/en/latest/)

class OpenMVS {
    fs::path reconstruction_dir;
    fs::path interface_mvs_path;
    fs::path densify_path;
    fs::path mesh_reconstruction_path;
    fs::path mesh_refinement_path;
    fs::path mesh_texture_path;

    void convert_from_nvm_to_mvs();

    void densify_point_cloud();

    void remove_nan_values(MVS::PointCloud::PointArr & cloud);

    void remove_nan_points();

    void reconstruct_mesh(float const d);

    void refining_mesh();

    std::string double_to_string(double val);

    void resize_mesh(double ratio, double agressiveness);

    fs::path texturing_mesh(double const ratio);

    void centering_textured_mesh(fs::path const & textured_mesh_path);

public:
    explicit OpenMVS(fs::path const & dir);

    void build_model_from_sparse_point_cloud();
};

#endif //RECONSTRUCTION_OPENMVS_H

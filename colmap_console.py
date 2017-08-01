from multiprocessing import Process
from shutil import copyfile
import os
import subprocess
import sys
import pymesh
import argparse
import time
import cv2
import numpy as np

# Compiling from sources:
# 1) OpenMVS (https://github.com/cdcseacave/openMVS/wiki/Building)
# 2) COLMAP (https://colmap.github.io/install.html#build-from-source)
# 3) PyMesh (https://github.com/qnzhou/PyMesh#build)

# For OpenMVS and others using:
# 1) Eigen 3.2.10 (3.3._ doesn't works)
# 2) Ceres-solver (http://ceres-solver.org/installation.html)

COLMAP_BIN = "/usr/local/bin"
OPENMVS_BIN = "/usr/local/bin/OpenMVS"
REMOVE_NAN = "/home/user/PycharmProjects/Reconstruction"
WORKING_PATH = "/result"
SEQUENTIAL_PATH = "/sequential_matching"
EXHAUSTIVE_PATH = "/exhaustive_matching"
DATABASE_PATH = "/database.db"


# Print highlighting
class Colors:
    def __init__(self):
        None
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    WARNING = '\033[93m'
    ENDC = '\033[0m'
    FAIL = '\033[91m'


def print_header(some_str):
    print Colors.HEADER + some_str + Colors.ENDC


# Object detection and filling background with black pixels (http://www.codepasta.com/site/vision/segmentation/)
def get_sobel(color_channel):
    sobel_dx = cv2.Sobel(color_channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)  # OX gradient
    sobel_dy = cv2.Sobel(color_channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)  # OY gradient
    sobel = np.hypot(sobel_dx, sobel_dy)  # get the magnitude of gradients combined
    return sobel


def find_significant_contours(sobel_8u):
    # The thing to understand here is "heirarchical" contours.
    # What that means is, any contour (c1) enclosed inside another contour (c2) is treated as a "child" of c2.
    # And contours can be nested to more than one level (So the structure is like a tree).
    # OpenCV returns the tree as a flat array though; with each tuple containing the index to the parent contour.
    image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    # Next we remove any contour that doesn't take up at least 5% of the image in area.
    significant = []
    # If contour isn't covering 5% of total area of image then it probably is too small
    too_small = sobel_8u.size * 5 / 100
    for tupl in level1:
        contour = contours[tupl[0]]
        area = cv2.contourArea(contour)
        if area > too_small:
            # cv2.drawContours(img, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant]


def scale_image(img):
    # 2560x1920 - 5 Megapixel - consume a lot of time
    # 2240x1680 - 4 Megapixel - consume a lot of time
    # 2048x1536 - 3 Megapixel - consume more time than
    # 1920x1440 ~ 2.5 Mpx - compromise between spent time and good quality
    # 1600x1200 - 2 Megapixel - get bad results

    height, width = img.shape[:2]
    min_size = min(height, width)
    max_size = max(height, width)
    # Scale big image to 3:4 format: 1920x1440 (WxH)
    if min_size > 1440 or max_size > 1920:
        if height > width:
            h_multi = 1920.0 / height
            w_multi = 1440.0 / width
        else:
            h_multi = 1440.0 / height
            w_multi = 1920.0 / width
        img = cv2.resize(img, None, fx=w_multi, fy=h_multi, interpolation=cv2.INTER_CUBIC)
    return img


def object_detection(path, write_path):
    img = cv2.imread(path)
    img = scale_image(img)

    # STEP 1. Edge detection
    blurred = cv2.GaussianBlur(img, (9, 9), 0)  # Remove noise
    # Edge operator
    # Since we are dealing with color images,
    # the edge detction needs to be run on each color channel and then they need to be combined.
    # The way I am doing that is by finding the max intesity from among the R, G and B edges.
    # I've tried using average of the R,G,B edges, however max seems to give better results.
    sobel = np.max(
        np.array([get_sobel(blurred[:, :, 0]), get_sobel(blurred[:, :, 1]), get_sobel(blurred[:, :, 2])]), axis=0
    )

    # STEP 2. Noise removing
    # Noise reduction trick, from http://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l182
    mean = np.mean(sobel)

    # Zero any values less than mean. This reduces a lot of noise.
    sobel[sobel <= mean] = 0
    sobel[sobel > 255] = 255

    sobel_8u = np.asarray(sobel, np.uint8)

    # STEP 3. Contour detection
    significant = find_significant_contours(sobel_8u)

    # STEP 4. Background removing by creating a mask to fill the contours.
    # Mask
    mask = sobel.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)

    # Invert mask
    mask = np.logical_not(mask)

    # Finally remove the background
    img[mask] = 0

    fname = path.split('/')[-1]

    # cv2.imshow("Original", np.hstack([img, im]))
    # cv2.waitKey(0)
    cv2.imwrite(write_path + fname, img)
    print (path)


def image_processing(write_path, process):
    image_path = get_parent_dir(get_parent_dir(write_path))
    if process:
        for filename in sorted(os.listdir(image_path)):
            if ".jpg" in filename.lower():
                curr_image = os.path.join(os.path.abspath(image_path), filename)
                object_detection(curr_image, write_path + "/")


# Arguments parsing and create directory tree
def path_existence(input_dir):
    if not os.path.exists(input_dir):
        import sys
        print("Directory " + input_dir + " doesn't exists!")
        sys.exit("Please set correct full path to directory which contains the pictures set.")


def parse_args():
    # ARGS
    parser = argparse.ArgumentParser(description="Photogrammetry reconstruction with OpenMVG")
    # parser.print_help()
    parser.add_argument('input_dir', nargs='+', type=str, help="the directory which contains the pictures set.")

    # Parse args and check dir existence
    args = parser.parse_args()
    input_dir = os.path.abspath("".join(args.input_dir))
    path_existence(input_dir)

    return input_dir


def create_directory(full_path):
    if not os.path.exists(full_path):
        os.mkdir(full_path)


def get_parent_dir(directory):
    return os.path.dirname(directory)


def create_dir_structure(input_dir):
    create_directory(input_dir + WORKING_PATH)
    write_path = input_dir + WORKING_PATH
    create_directory(write_path + SEQUENTIAL_PATH)
    create_directory(write_path + EXHAUSTIVE_PATH)
    write_path += "/images"
    create_directory(write_path)
    return write_path


# Process creating and aborting
def check_process_ending(process):
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print Colors.FAIL + stdout + Colors.ENDC
        print "3D reconstruction failed!"
        print Colors.FAIL + stderr + Colors.ENDC
        sys.exit(-1)
    else:
        print Colors.OKBLUE + "Success" + Colors.ENDC


def start_process(args):
    process = subprocess.Popen(args, stderr=subprocess.PIPE)  # stdout=subprocess.PIPE,
    check_process_ending(process)


# COLMAP SFM pipeline (https://colmap.github.io/tutorial.html#structure-from-motion)
# 1. Feature extraction.
# 2. Matching (sequential or exhaustive)
# 3. Sparse reconstruction (camera positions, sparse point cloud, 2D-3D projections)
# 4. Image undistortion for correct dense reconstruction
# 5. Convert COLMAP data to NVM format. Then convert NVM to MVS format.
def extract_features(input_directory):
    print_header("1. Extract features")
    database_path = input_directory + DATABASE_PATH
    args = [os.path.join(COLMAP_BIN, "feature_extractor"),
            "--image_path", input_directory,
            "--database_path", database_path,
            "--ImageReader.single_camera", "1",
            "--SiftCPUExtraction.num_threads", "8",
            "--use_gpu", "1"
            ]
    start_process(args)
    input_directory = get_parent_dir(input_directory)
    copyfile(database_path, input_directory + SEQUENTIAL_PATH + DATABASE_PATH)
    copyfile(database_path, input_directory + EXHAUSTIVE_PATH + DATABASE_PATH)


def sequential_matcher(input_directory):
    print_header("2. Sequential matcher")
    working_directory = input_directory + SEQUENTIAL_PATH
    args = [os.path.join(COLMAP_BIN, "sequential_matcher"),
            "--database_path", working_directory + DATABASE_PATH,
            "--SiftMatching.num_threads", "8"
            ]
    start_process(args)
    return working_directory


def exhaustive_matcher(input_directory):
    print_header("2. Exhaustive matcher")
    working_directory = input_directory + EXHAUSTIVE_PATH
    args = [os.path.join(COLMAP_BIN, "exhaustive_matcher"),  # sequential_matcher
            "--database_path", working_directory + DATABASE_PATH,
            "--SiftMatching.num_threads", "8"
            ]
    start_process(args)
    return working_directory


def sparse_reconstruction(input_directory):
    print_header("3. Mapper")
    export_path = input_directory + "/sparse"
    create_directory(export_path)
    args = [os.path.join(COLMAP_BIN, "mapper"),
            "--database_path", input_directory + DATABASE_PATH,
            "--image_path", input_directory,
            "--export_path", export_path,
            "--Mapper.num_threads", "8"
            ]
    start_process(args)


def image_undistorter(image_path, input_directory):
    print_header("4. Image undistorter")
    output_path = input_directory + "/dense"
    create_directory(output_path)
    args = [os.path.join(COLMAP_BIN, "image_undistorter"),
            "--image_path", image_path,
            "--input_path", input_directory + "/sparse/0",
            "--output_path", output_path,
            "--output_type", "COLMAP"
            ]
    start_process(args)


def model_converter_from_colmap_to_nvm(input_directory):
    print_header("5. Model converter")
    undistorted_images_path = input_directory + "/dense/images"
    nvm_model_path = undistorted_images_path + "/model.nvm"
    args = [os.path.join(COLMAP_BIN, "model_converter"),
            "--input_path", input_directory + "/sparse/0",
            "--output_path", nvm_model_path,
            "--output_type", "nvm"
            ]
    start_process(args)
    convert_from_nvm_to_mvs(nvm_model_path)
    return undistorted_images_path


def convert_from_nvm_to_mvs(path_to_nvm_file):
    print_header("6. Convert " + path_to_nvm_file + " to " + path_to_nvm_file.split('.')[0] + ".mvs")
    working_folder = path_to_nvm_file.rsplit("/", 1)[0]
    args = [os.path.join(OPENMVS_BIN, "InterfaceVisualSFM"),
            "-i", path_to_nvm_file,
            "-w", working_folder,
            "-o", working_folder + "/scene.mvs",
            ]
    start_process(args)


# OpenMVS pipeline (https://github.com/cdcseacave/openMVS/wiki/Usage)
# It has 4 modules (https://github.com/cdcseacave/openMVS/wiki/Modules)
# 1. Sparse pointcloud densifying (http://www.connellybarnes.com/work/publications/2011_patchmatch_cacm.pdf)
#           (PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing C. Barnes et al. 2009).

# 2. Remove NAN points after densifying. (https://github.com/cdcseacave/openMVS/wiki/Interface)
#           (Using MVS interface)

# 3. Mesh reconstruction (https://www.hindawi.com/journals/isrn/2014/798595/)
#           (Exploiting Visibility Information in Surface Reconstruction
#               to Preserve Weakly Supported Surfaces M. Jancosek et al. 2014).
#
#   M. Jancosek dissertation: Large Scale Surface Reconstruction based on Point Visibility
#       (https://dspace.cvut.cz/bitstream/handle/10467/60872/Disertace_Jancosek_2014.pdf?sequence=1)
#
# 4. Mesh refinement (http://sci-hub.cc/http://ieeexplore.ieee.org/document/5989831/)
#           (High Accuracy and Visibility-Consistent Dense Multiview Stereo HH. Vu et al. 2012).
#
# 5. Mesh texturing (https://pdfs.semanticscholar.org/c8e6/eefd01b17489d38c355cf21dd492cbd02dab.pdf)
#           (Let There Be Color! - Large-Scale Texturing of 3D Reconstructions M. Waechter et al. 2014).
#
# 6. Remeshing and retexturing (Pymesh + Mesh texturing) (http://pymesh.readthedocs.io/en/latest/)

def densify_point_cloud(reconstruction_dir):
    print_header("6. Densify point cloud")
    args = [os.path.join(OPENMVS_BIN, "DensifyPointCloud"),
            "-i", reconstruction_dir + "/scene.mvs",
            "-w", reconstruction_dir,
            "-o", reconstruction_dir + "/scene_dense.mvs",
            "--process-priority", "1",
            "--resolution-level", "1"
            ]
    start_process(args)


def remove_nan_points(reconstruction_dir):
    print_header("6. Removing NAN values from dense point cloud.")
    args = [os.path.join(REMOVE_NAN, "RemoveNan"),
            reconstruction_dir + "/scene_dense.mvs"
            ]
    start_process(args)


def reconstruct_mesh(reconstruction_dir):
    print_header("7. Reconstruct the mesh")
    args = [os.path.join(OPENMVS_BIN, "ReconstructMesh"),
            "-i", reconstruction_dir + "/scene_dense.mvs",
            "-w", reconstruction_dir,
            "--process-priority", "1",
            "-d", "7",
            "--thickness-factor", "1.0",
            "--quality-factor", "2.5",
            "--close-holes", "30",
            "--smooth", "3"
            ]
    start_process(args)


def refine_mesh(reconstruction_dir):
    print_header("7. Refine the mesh")
    args = [os.path.join(OPENMVS_BIN, "RefineMesh"),
            "-i", reconstruction_dir + "/scene_dense_mesh.mvs",
            "--process-priority", "1",
            "--resolution-level", "0",
            "--ensure-edge-size", "2",
            "--close-holes", "30",
            "-w", reconstruction_dir
            ]
    start_process(args)


def texture_mesh(reconstruction_dir, length):
    print_header("8. Texture the remeshed model")
    if length == 0:
        mesh_file_name = reconstruction_dir + "/scene_dense_mesh_refine.ply"
    else:
        mesh_file_name = reconstruction_dir + "/remesh_" + str(length) + ".ply"
    output_path = get_parent_dir(get_parent_dir(reconstruction_dir))
    args = [os.path.join(OPENMVS_BIN, "TextureMesh"),
            "-i", reconstruction_dir + "/scene_dense.mvs",
            "-o", output_path + "/scene_texture_" + str(length) + ".mvs",
            "-w", reconstruction_dir,
            "--process-priority", "1",
            "--export-type", "obj",
            "--mesh-file", mesh_file_name
            ]
    start_process(args)


def remeshing_and_texture(reconstruction_dir):
    texture_mesh(reconstruction_dir, 0)
    print_header("Remeshing the model")
    length = 1.0
    while 1:
        print("Trying to resize mesh and texture. Current edge collapse threshold " + str(length) + ".")
        start = time.time()
        mesh = pymesh.load_mesh(reconstruction_dir + "/scene_dense_mesh_refine.ply")
        mesh, info = pymesh.collapse_short_edges(mesh, rel_threshold=float(length))
        if len(mesh.vertices) > 1000 and length < 4:
            pymesh.save_mesh(reconstruction_dir + "/remesh_" + str(length) + ".ply", mesh)
            end = time.time()
            print("Elapsed time: %d sec" % (end - start) + "\n")
            texture_mesh(reconstruction_dir, length)
            length += 0.5
        else:
            print ("Can't texture mapping on resized mesh with threshold " + str(length) + "!")
            return -1


# Modules union to pipeline
def run_feature_matcher(working_directory, bool_sequential):
    working_directory = get_parent_dir(working_directory)
    if bool_sequential:
        current_dir = sequential_matcher(working_directory)
    else:
        current_dir = exhaustive_matcher(working_directory)
    return current_dir


def run_sfm(working_directory, bool_sequential):
    matches_directory = run_feature_matcher(working_directory, bool_sequential)
    sparse_reconstruction(matches_directory)
    image_undistorter(working_directory, matches_directory)
    reconstruction_directory = model_converter_from_colmap_to_nvm(matches_directory)
    return reconstruction_directory


def run_mvs(reconstruction_dir):
    densify_point_cloud(reconstruction_dir)
    remove_nan_points(reconstruction_dir)
    reconstruct_mesh(reconstruction_dir)
    refine_mesh(reconstruction_dir)
    remeshing_and_texture(reconstruction_dir)


def sfm_mvs_pipeline(working_directory, bool_sequential):
    reconstruction_directory = run_sfm(working_directory, bool_sequential)
    run_mvs(reconstruction_directory)


# MAIN

# parse arguments and check path existence
start_time = time.time()
inputDir = parse_args()
workingDir = create_dir_structure(inputDir)
image_processing(workingDir, True)  # change dir to dir with processed images (inputDir/binary/)
extract_features(workingDir)

curr_time = time.time()
p = Process(target=sfm_mvs_pipeline, args=(workingDir, 1,))
p.start()
p.join()
end_time = time.time()
print("--- %s seconds ---" % round((time.time() - start_time), 2))

p1 = Process(target=sfm_mvs_pipeline, args=(workingDir, 0,))
p1.start()
p1.join()
print("---%s seconds ---" % round((time.time() - start_time - (end_time - curr_time)), 2))


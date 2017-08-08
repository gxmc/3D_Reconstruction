//
// Created by user on 8/4/17.
//
#include "image_processing.h"

// Resize image to to 3:4 format: 1920x1440 (WxH)
cv::Mat ImageProcessing::scale_image(cv::Mat & img) {
    // 2560x1920 - 5 Megapixel - consume a lot of time
    // 2240x1680 - 4 Megapixel - consume a lot of time
    // 2048x1536 - 3 Megapixel - consume more time than
    // 1920x1440 ~ 2.5 Mpx - compromise between spent time and good quality
    // 1600x1200 - 2 Megapixel - get bad results
    int width = img.cols;
    int height = img.rows;
    int min_size = std::min(height, width);
    int max_size = std::max(height, width);
    // Scale image to 1920x1440
    if ((min_size > 1440) || (max_size > 1920)) {
        double h_multi = 0.0, w_multi = 0.0;
        if (height > width) {
            h_multi = 1920.0 / height;
            w_multi = 1440.0 / width;
        } else {
            h_multi = 1440.0 / height;
            w_multi = 1920.0 / width;
        }
        cv::resize(img, img, cv::Size(0, 0), w_multi, h_multi, cv::INTER_CUBIC);
    }
    return img;
}

// Get the magnitude of gradients
cv::Mat ImageProcessing::calc_magnitude(cv::Mat const & gx, cv::Mat const & gy) {
    cv::Mat magnitude(gx.rows, gx.cols, CV_16S);
    for (auto i = 0; i < gx.rows; ++i) {
        for (auto j = 0; j < gx.cols; ++j) {
            magnitude.at<short>(i, j) = short(
                    sqrt(gx.at<short>(i, j) * gx.at<short>(i, j) +  gy.at<short>(i, j) * gy.at<short>(i, j))
            );
        }
    }
    return magnitude;
}

// Object detection and filling background with black pixels (http://www.codepasta.com/site/vision/segmentation/)
cv::Mat ImageProcessing::get_sobel(cv::Mat const & channel) {
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y;
    int scale = 1;
    int delta = 0;
    int ddepth =  CV_16S;
    cv::Sobel(channel, grad_x, ddepth, 1, 0, 3, 1, delta, cv::BORDER_DEFAULT ); // OX gradient
    cv::Sobel(channel, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT ); // OY gradient
    return calc_magnitude(grad_x, grad_y); // get the magnitude of gradients combined
}

short ImageProcessing::max(short const x, short const y, short const z) {
    return std::max(std::max(x, y), z);
}

// Get max intensity from 3 BGR channels and calculate intensity mean value
cv::Mat ImageProcessing::get_max_intensity(cv::Mat const & b, cv::Mat const & g, cv::Mat const & r, long & mean_value) {
    cv::Mat magnitude(b.rows, b.cols, CV_16S);
    for (auto i = 0; i < b.rows; ++i) {
        for (auto j = 0; j < b.cols; ++j) {
            short max_intensity = max(b.at<short>(i, j), g.at<short>(i, j), g.at<short>(i, j));
            magnitude.at<short>(i, j) = max_intensity;
            mean_value += max_intensity;
        }
    }
    mean_value = long(float(mean_value) / (b.rows * b.cols));
    return magnitude;
}

// Zero any values less than mean_value. This reduces a lot of noise.
void ImageProcessing::remove_noise(cv::Mat & magnitude, long & mean_value) {
    for (auto i = 0; i < magnitude.rows; ++i) {
        for (auto j = 0; j < magnitude.cols; ++j) {
            short current_value = magnitude.at<short>(i, j);
            if (current_value <= mean_value) {
                magnitude.at<short>(i, j) = 0;
            }
            if (current_value > 255) {
                magnitude.at<short>(i, j) = 255;
            }
        }
    }
}

void ImageProcessing::find_significant_contours(cv::Mat const & magnitude, Vector <Vector<cv::Point>> & points) {
    // The thing to understand here is "heirarchical" contours.
    // What that means is, any contour (c1) enclosed inside another contour (c2) is treated as a "child" of c2.
    // And contours can be nested to more than one level (So the structure is like a tree).
    // OpenCV returns the tree as a flat array though; with each tuple containing the index to the parent contour.
    std::vector <std::vector <cv::Point>> contours;
    std::vector <cv::Vec4i> hierarchy;
    findContours(magnitude, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    // Find level 1 contours
    std::vector<contour_with_number> level1;
    level1.reserve(hierarchy.size());
    for (auto i = 0; i < hierarchy.size(); ++i) {
        // Each array is in format (Next, Prev, First child, Parent)
        // Filter the ones without parent
        if (hierarchy[i][3] == -1) {
            contour_with_number new_contour(i, hierarchy[i]);
            level1.push_back(new_contour);
        }
    }

    // From among them, find the contours with large surface area.
    // Next we remove any contour that doesn't take up at least 5% of the image in area.
    std::vector<contour_with_area> significant;
    significant.reserve(level1.size());

    // If contour isn't covering 5% of total area of image then it probably is too small
    double too_small = magnitude.rows * magnitude.cols * 5 / 100.0;
    for (auto tuple : level1) {
        Vector <Vector <cv::Point>> contour(1);
        contour[0] = contours[tuple.first];
        int area = int(contourArea(contour[0]));
        if (area > too_small) {
//                drawContours(img, contour, 0, (0, 255, 0), 2, cv::LINE_AA);
            significant.push_back(std::make_pair(contour[0], area));
        }
    }

    // Sort significant contours per area
    std::sort(significant.begin(), significant.end(),
              [](contour_with_area a, contour_with_area b) {
                  return a.second < b.second;
              }
    );

    // Get significant contours without their area
    points.reserve(significant.size());
    for (auto tuple: significant) {
        points.push_back(tuple.first);
    }
}

// Apply background-mask to image
void ImageProcessing::apply_mask(cv::Mat & image, cv::Mat const & mask) {
    // Fill background with zero values
    for (auto i = 0; i < mask.rows; ++i) {
        for (auto j = 0; j < mask.cols; ++j) {
            if (!mask.at<short>(i, j) > 0) {
                cv::Vec3b bgrPixel = image.at<cv::Vec3b>(i, j);
                bgrPixel[0] = 0;
                bgrPixel[1] = 0;
                bgrPixel[2] = 0;
                image.at<cv::Vec3b>(i, j) = bgrPixel;
            }
        }
    }
}

// Detect object on image and fill background with black color
void ImageProcessing::object_detection(cv::Mat & img, std::string const & image_filename, fs::path const & write_path) {
    img = scale_image(img);

    // STEP 1. Edge detection. Blurring image
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(9, 9), 0);
    // Since we are dealing with color images, the edge detection needs to be run on each color channel
    // and then they need to be combined.
    // The way I am doing that is by finding the max intensity from among the R, G and B edges.
    // I've tried using average of the R,G,B edges, however max seems to give better results.
    std::vector<cv::Mat> channels(3);
    split(blurred, channels);
    cv::Mat edges;
    for (int i = 0; i < channels.size(); ++i) {
        channels[i] = get_sobel(channels[i]);
    }
    long mean_value = 0;
    edges = get_max_intensity(channels[0], channels[1], channels[2], mean_value);

    // STEP 2. Noise removing
    // Noise reduction trick, from http://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l182
    remove_noise(edges, mean_value);
    cv::Mat scaled;
    // Convert converts FP32 (single precision floating point) from/to FP16 (half precision floating point).
    // Now we can show image
    convertScaleAbs(edges, scaled);

    // STEP 3. Significant contours detection
    Vector< Vector <cv::Point>> significant;
    find_significant_contours(scaled, significant);

    // STEP 4. Background removing by creating a mask to fill the contours.
    // Mask
    cv::Mat mask(img.rows, img.cols, CV_16S, double(0));
    cv::fillPoly(mask, significant, 255);

    // Finally remove the background
    apply_mask(img, mask);


    // Save image
    std::string output_image_path = write_path.string() + "/" + image_filename.substr(0, image_filename.size() - 3) + "jpg";
//    std::cout << write_path.c_str() << std::endl;
    imwrite(output_image_path, img);
}

// Create directory tree
fs::path ImageProcessing::create_dir_structure(fs::path const & path) {
    fs::path write_path = path / local_path::WORKING_PATH;
    fs::create_directory(write_path);
    fs::create_directory(write_path / local_path::SEQUENTIAL_PATH);
    fs::create_directory(write_path / local_path::EXHAUSTIVE_PATH);
    write_path /= "images";
    fs::create_directory(write_path);
    return write_path;
}

ImageProcessing::ImageProcessing(fs::path & path) : working_dir(create_dir_structure(path)) {};

// Read files, try open them as images and object detection
void ImageProcessing::start() {
    fs::path image_path = working_dir.parent_path().parent_path();
    std::cout << image_path << std::endl;
    for (auto &p : fs::directory_iterator(image_path)) {
        if (p.path().has_extension()) {
            cv::Mat img = cv::imread(p.path().string());
            if (!img.data) {
                std::cout << "Can't open " + p.path().string() + " as image." << std::endl;
            } else {
                object_detection(img, p.path().filename(), working_dir);
            }
        }
    }
}

std::string ImageProcessing::get_working_dir() const {
    return working_dir.c_str();
}

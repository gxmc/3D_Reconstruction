//
// Created by user on 8/4/17.
//

#ifndef RECONSTRUCTION_IMAGE_PROCESSING_H
#define RECONSTRUCTION_IMAGE_PROCESSING_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "utils.h"


class ImageProcessing {
    template<class T>
    using Vector = std::vector<T>;

    typedef std::pair <int, cv::Vec4i> contour_with_number;
    typedef std::pair <Vector<cv::Point>, int> contour_with_area;

    fs::path working_dir;

    // Resize image to to 3:4 format: 1920x1440 (WxH)
    cv::Mat scale_image(cv::Mat & img);

    // Get the magnitude of gradients
    cv::Mat calc_magnitude(cv::Mat const & gx, cv::Mat const & gy);

    // Object detection and filling background with black pixels (http://www.codepasta.com/site/vision/segmentation/)
    cv::Mat get_sobel(cv::Mat const & channel);

    short max(short x, short y, short z);

    // Get max intensity from 3 BGR channels and calculate intensity mean value
    cv::Mat get_max_intensity(cv::Mat const & b, cv::Mat const & g, cv::Mat const & r, long & mean_value);

    // Zero any values less than mean_value. This reduces a lot of noise.
    void remove_noise(cv::Mat & magnitude, long & mean_value);

    void find_significant_contours(cv::Mat const & magnitude, Vector <Vector<cv::Point>> & points);

    // Apply background-mask to image
    void apply_mask(cv::Mat & image, cv::Mat const & mask);

    // Detect object on image and fill background with black color
    void object_detection(cv::Mat & img, std::string const & image_filename, fs::path const & write_path);

    // Create directory tree
    fs::path create_dir_structure(fs::path const & path);
public:
    explicit ImageProcessing(fs::path & path);

    // Read files, try open them as images and object detection
    void start();

    std::string get_working_dir() const;
};

#endif //RECONSTRUCTION_IMAGE_PROCESSING_H

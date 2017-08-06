//
// Created by user on 8/4/17.
//

#ifndef RECONSTRUCTION_IMAGE_PROCESSING_H
#define RECONSTRUCTION_IMAGE_PROCESSING_H

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "utils.h"


class ImageProcessing {
    template<class T>
    using Vector = std::vector<T>;

    typedef std::pair <int, cv::Vec4i> contour_with_number;
    typedef std::pair <Vector<cv::Point>, int> contour_with_area;

    fs::path working_dir;

    cv::Mat calc_magnitude(cv::Mat const & dx, cv::Mat const & dy);

    short biggest(short x, short y, short z);

    cv::Mat get_max_intensity(cv::Mat const & b, cv::Mat const & g, cv::Mat const & r, long & mean_value);

    void remove_noise(cv::Mat & magnitude, long & mean_value);

    cv::Mat get_sobel(cv::Mat const & channel);

    void find_significant_contours(cv::Mat const & magnitude, Vector <Vector<cv::Point>> & points);

    void apply_mask(cv::Mat & image, cv::Mat const & mask);

    cv::Mat scale_image(cv::Mat & img);

    void object_detection(fs::path const & image_path, fs::path write_path);

    fs::path create_dir_structure(fs::path const & path);

public:
    explicit ImageProcessing(fs::path & path);

    void start();

    std::string get_working_dir() const;
};

#endif //RECONSTRUCTION_IMAGE_PROCESSING_H

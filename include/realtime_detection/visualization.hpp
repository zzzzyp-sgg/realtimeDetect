/**
 * @file   visualization.hpp
 * @brief  draw detection result on image
 * @author Yipeng Zhao
*/

#ifndef VISUALIZATION_HPP
#define VISUALIZATION_HPP

#include "yolox.hpp"
#include <fstream>

class Visualization
{
public:
    bool setLabelFile(const std::string &label_input);

    cv::Mat drawObjects(const cv::Mat &in, const ObjectArray &objects);

public:
    std::map<int, std::string> label_map_;
};

#endif  // VISUALIZATION_HPP
#include "realtime_detection/visualization.hpp"
#include "realtime_detection/color.hpp"
#include <ros/ros.h>

bool Visualization::setLabelFile(const std::string &label_input)
{
    std::ifstream label_file(label_input);
    if (!label_file.is_open()) {
        ROS_ERROR_STREAM("Could not open label file. " << label_input);
        return false;
    }
    int label_index{};
    std::string label;
    while (std::getline(label_file, label)) {
    std::transform(
        label.begin(), label.end(), label.begin(), [](auto c) { return std::toupper(c); });
        label_map_.insert({label_index, label});
        ++label_index;
    }
    return true;
}

cv::Mat Visualization::drawObjects(const cv::Mat &in, const ObjectArray &objects)
{
    if (objects.empty())
    {
        ROS_WARN_STREAM("Cannot detect any object!");
    }

    cv::Mat output_img = in;

    for (const auto & object : objects) {
        // color
        float* color_f = _COLORS[object.type];
        std::vector<int> color = { static_cast<int>(color_f[0] * 255), static_cast<int>(color_f[1] * 255), static_cast<int>(color_f[2] * 255) };

        // text
        std::string text = label_map_[object.type] + ":" + std::to_string(object.score * 100) + "%";
        cv::Scalar txt_color = ((color_f[0] + color_f[1] + color_f[2]) > 0.5) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
        int font = cv::FONT_HERSHEY_SIMPLEX;
        int baseline = 0;
        cv::Size txt_size = cv::getTextSize(text, font, 0.4, 1, &baseline);
        
        const auto left = object.x_offset;
        const auto top = object.y_offset;
        const auto right = std::clamp(left + object.width, 0, output_img.cols);
        const auto bottom = std::clamp(top + object.height, 0, output_img.rows);
        cv::rectangle(
        output_img, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 3, 8, 0);

        // text bg
        std::vector<int> txt_bk_color = { static_cast<int>(color_f[0] * 255 * 0.7), static_cast<int>(color_f[1] * 255 * 0.7), static_cast<int>(color_f[2] * 255 * 0.7) };
        cv::rectangle(
            output_img,
            cv::Point(left, top + 1),
            cv::Point(left + txt_size.width + 1, top + int(1.5 * txt_size.height)),
            cv::Scalar(txt_bk_color[0], txt_bk_color[1], txt_bk_color[2]),
            -1
        );

        cv::putText(output_img, text, cv::Point(left, top + txt_size.height), font, 0.4, txt_color, 1);
    }

    return output_img;
}
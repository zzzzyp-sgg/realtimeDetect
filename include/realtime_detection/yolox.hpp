/**
 * @file   yolox.hpp
 * @brief  Deploying YOLOx model and do interpreter on QCS6490
 * @author Yopeng Zhao 
 * @ref    https://autowarefoundation.github.io/autoware.universe/main/perception/tensorrt_yolox/
*/

#ifndef YOLOX_HPP
#define YOLOX_HPP

#include <iostream>
#include <memory>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <ros/ros.h>

// opencv
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/dnn/dnn.hpp"

#include "aidlux/aidlite/aidlite.hpp"

#define NUM_CLASSES 80

using namespace Aidlux::Aidlite;

struct GridAndStride
{
  int grid0;
  int grid1;
  int stride;
};

struct Object
{
  int32_t x_offset;
  int32_t y_offset;
  int32_t height;
  int32_t width;
  float score;
  int32_t type;
};

using ObjectArray = std::vector<Object>;
using ObjectArrays = std::vector<ObjectArray>;

class YoloX
{
public:
    YoloX(const std::string &model_path, const int &resolution, 
      const float &score_threshold, const float &nms_threshold, bool quantify = false);

    bool doInference(const cv::Mat &images, ObjectArray &objects);

    bool isInited() const
    {
        return initialized_;
    }
private:
    /// @brief init interpreter
    bool init();

    /// @brief preprocess the image
    void preProcess(const cv::Mat &images, cv::Mat &input_data);

    /// @brief generate grids and strides for post process
    void generateGridsAndStride(const int target_w, const int target_h, std::vector<int> & strides,
      std::vector<GridAndStride> & grid_strides) const;

    /// @brief  generate proposals
    void generateYoloxProposals(
      std::vector<GridAndStride> grid_strides, float * feat_blob, float prob_threshold,
      ObjectArray & objects) const;

    /// @brief  Sort candidate boxes by confidence level 
    void qsortDescentInplace(ObjectArray & faceobjects, int left, int right) const;
    inline void qsortDescentInplace(ObjectArray & objects) const
    {
      if (objects.empty()) {
        return;
      }
      qsortDescentInplace(objects, 0, objects.size() - 1);
    }

    /// @brief  get intersection area
    inline float intersectionArea(const Object & a, const Object & b) const
    {
      cv::Rect a_rect(a.x_offset, a.y_offset, a.width, a.height);
      cv::Rect b_rect(b.x_offset, b.y_offset, b.width, b.height);
      cv::Rect_<float> inter = a_rect & b_rect;
      return inter.area();
    }

    /// @brief  nms sorted
    void nmsSortedBboxes(
      const ObjectArray & faceobjects, std::vector<int> & picked, float nms_threshold) const;

    /// @brief  post-process
    void postProcess(
      float * prob, ObjectArray & objects, float scale, cv::Size & img_size) const;                   

    int resolution_;
    // std::shared_ptr<Context> context_;
    std::shared_ptr<Interpreter> interpreter_;
    std::chrono::time_point<std::chrono::system_clock> t_b_, t_e_;

    float scale_;
    float score_threshold_;
    float nms_threshold_;
    bool initialized_;
};

#endif
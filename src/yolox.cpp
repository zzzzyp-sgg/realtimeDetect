#include "realtime_detection/yolox.hpp"

YoloX::YoloX(const std::string &model_path, const int &resolution, 
      const float &score_threshold, const float &nms_threshold, bool quantify)
        : resolution_(resolution), score_threshold_(score_threshold), nms_threshold_(nms_threshold), initialized_(false)
{
    // create Model and set params
    Model* model  = Model::create_instance(model_path);
    assert(model != nullptr);
    std::vector<std::vector<uint32_t>> input_shapes = {{1, (uint32_t)resolution, (uint32_t)resolution, 3}};
    uint32_t output_dim = pow((resolution / 32), 2) + 
                            pow((resolution / 16), 2) + pow((resolution / 8), 2);
    assert(output_dim == 3549);
    std::vector<std::vector<uint32_t>> output_shapes = {{1, output_dim, 85}};
    model->set_model_properties(input_shapes, DataType::TYPE_FLOAT32,
                                output_shapes, DataType::TYPE_FLOAT32);

    // create Config and set params
    Config* config = Config::create_instance();
    if (config == nullptr)
    {
        ROS_FATAL_STREAM("Create Config failed!");
    }
    config->framework_type = FrameworkType::TYPE_TFLITE;
    config->accelerate_type = AccelerateType::TYPE_GPU;
    config->is_quantify_model = quantify;

    interpreter_ = InterpreterBuilder::build_interpretper_from_model_and_config(model, config);
    
    initialized_ = init();
}

bool YoloX::init()
{
    // initialize
    int result = interpreter_->init();
    assert(result == 0);
    // load model
    result = interpreter_->load_model();
    if (result != 0)
    {
        ROS_FATAL_STREAM("interpreter load model failed!");
    }

    return true;
}

void YoloX::preProcess(const cv::Mat &img, cv::Mat &input_data)
{
    cv::Mat dst_img;
    cv::cvtColor(img, dst_img, cv::COLOR_BGR2RGB);
    scale_ = std::min((float)resolution_ / img.cols, (float)resolution_ / img.rows);
    ROS_INFO_STREAM("scale: " << scale_);
    cv::Size scale_size = cv::Size(img.cols * scale_, img.rows * scale_);
    cv::resize(dst_img, dst_img, scale_size, 0, 0, cv::INTER_CUBIC);
    const auto bottom = resolution_ - dst_img.rows;
    const auto right = resolution_ - dst_img.cols;
    cv::copyMakeBorder(dst_img, dst_img, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});
    dst_img.convertTo(input_data, CV_32FC3);
}

bool YoloX::doInference(const cv::Mat &img, ObjectArray &objects)
{
    cv::Mat input_data;
    preProcess(img, input_data);

    int result = interpreter_->set_input_tensor(0, static_cast<void*>(input_data.data));

    t_b_ = std::chrono::system_clock::now();
    result = interpreter_->invoke() == 0;
    t_e_ = std::chrono::system_clock::now();
    auto time_cost = std::chrono::duration_cast<std::chrono::milliseconds>(t_e_ - t_b_);
    ROS_INFO_STREAM("invoke cost: " << time_cost.count() << "ms.");

    // TODO post-process
    float* out_data = nullptr;
    uint32_t output_tensor_length = 0;
    result = interpreter_->get_output_tensor(0, (void**)&out_data, &output_tensor_length);
    assert(result == 0);

    cv::Size img_size = img.size();
    postProcess(out_data, objects, scale_, img_size);
    ROS_INFO_STREAM("objects size: " << objects.size());

    return true;
}

void YoloX::postProcess(
      float * prob, ObjectArray & objects, float scale, cv::Size & img_size) const
{
    ObjectArray proposals;

    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generateGridsAndStride(resolution_, resolution_, strides, grid_strides);
    generateYoloxProposals(grid_strides, prob, score_threshold_, proposals);

    qsortDescentInplace(proposals);

    std::vector<int> picked;
    nmsSortedBboxes(proposals, picked, nms_threshold_);

    int count = static_cast<int>(picked.size());
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].x_offset) / scale;
        float y0 = (objects[i].y_offset) / scale;
        float x1 = (objects[i].x_offset + objects[i].width) / scale;
        float y1 = (objects[i].y_offset + objects[i].height) / scale;

        // clip
        x0 = std::clamp(x0, 0.f, static_cast<float>(img_size.width - 1));
        y0 = std::clamp(y0, 0.f, static_cast<float>(img_size.height - 1));
        x1 = std::clamp(x1, 0.f, static_cast<float>(img_size.width - 1));
        y1 = std::clamp(y1, 0.f, static_cast<float>(img_size.height - 1));

        objects[i].x_offset = x0;
        objects[i].y_offset = y0;
        objects[i].width = x1 - x0;
        objects[i].height = y1 - y0;
    }
}

void YoloX::generateGridsAndStride(const int target_w, const int target_h, std::vector<int> & strides,
                                    std::vector<GridAndStride> & grid_strides) const
{
    for (auto stride : strides) {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                grid_strides.push_back(GridAndStride{g0, g1, stride});
            }
        }
    }
}

void YoloX::generateYoloxProposals(
  std::vector<GridAndStride> grid_strides, float * feat_blob, float prob_threshold,
  ObjectArray & objects) const
{
  const int num_anchors = grid_strides.size();

  for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

    // yolox/models/yolo_head.py decode logic
    // To apply this logic, YOLOX head must output raw value
    // (i.e., `decode_in_inference` should be False)
    float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
    float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;
    float w = exp(feat_blob[basic_pos + 2]) * stride;
    float h = exp(feat_blob[basic_pos + 3]) * stride;
    float x0 = x_center - w * 0.5f;
    float y0 = y_center - h * 0.5f;

    float box_objectness = feat_blob[basic_pos + 4];
    for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++) {
      float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
      float box_prob = box_objectness * box_cls_score;
      if (box_prob > prob_threshold) {
        Object obj;
        obj.x_offset = x0;
        obj.y_offset = y0;
        obj.height = h;
        obj.width = w;
        obj.type = class_idx;
        obj.score = box_prob;

        objects.push_back(obj);
      }
    }  // class loop
  }    // point anchor loop
}

void YoloX::qsortDescentInplace(ObjectArray & faceobjects, int left, int right) const
{
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].score;

  while (i <= j) {
    while (faceobjects[i].score > p) {
      i++;
    }

    while (faceobjects[j].score < p) {
      j--;
    }

    if (i <= j) { // Put the big ones aside
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (left < j) {
        qsortDescentInplace(faceobjects, left, j);
      }
    }
#pragma omp section
    {
      if (i < right) {
        qsortDescentInplace(faceobjects, i, right);
      }
    }
  }
}

void YoloX::nmsSortedBboxes(
  const ObjectArray & faceobjects, std::vector<int> & picked, float nms_threshold) const
{
  picked.clear();
  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    cv::Rect rect(
      faceobjects[i].x_offset, faceobjects[i].y_offset, faceobjects[i].width,
      faceobjects[i].height);
    areas[i] = rect.area();
  }

  for (int i = 0; i < n; i++) {
    const Object & a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < static_cast<int>(picked.size()); j++) {
      const Object & b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersectionArea(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) {
        keep = 0;
      }
    }

    if (keep) {
      picked.push_back(i);
    }
  }
}
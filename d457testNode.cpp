/**
 * @file   d457testNode.cpp
 * @brief  ros node for RealsenseD457 detection test.
 * @author Yipeng Zhao
*/

#include "realtime_detection/yolox.hpp"
#include "realtime_detection/visualization.hpp"

#include <queue>
#include <thread>
#include <mutex> 

#include <cv_bridge/cv_bridge.h>
#include <ros/subscriber.h>
#include <ros/publisher.h>
#include <image_transport/image_transport.h>
#include <yaml-cpp/yaml.h>

struct YoloxConfig
{
    std::string input_model;
    std::string label_path;
    bool quantify;
    int resolution;
    double score_threshold;
    double nms_threshold;
};

std::queue<cv::Mat> ori_img_buf_;
std::mutex m_buf;
std::shared_ptr<YoloX> yolox_;
std::shared_ptr<Visualization> vis_;
image_transport::Publisher pub_img;

void readParameters(const std::string &yaml_file, YoloxConfig &config)
{
    YAML::Node yaml = YAML::LoadFile(yaml_file);

    config.input_model = yaml["input_model"].as<std::string>();
    config.label_path = yaml["label_path"].as<std::string>();
    int quant = yaml["quantify"].as<int>();
    config.quantify = quant ? true : false;
    config.resolution = yaml["resolution"].as<int>();
    config.score_threshold = yaml["score_threshold"].as<double>();
    config.nms_threshold = yaml["nms_threshold"].as<double>();
}

void image_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    
    if (yolox_->isInited()) ori_img_buf_.push(img);
}

void process()
{
    while (1)
    {
        cv::Mat input_img;
        m_buf.lock();
        if (ori_img_buf_.empty() || !yolox_->isInited())
        {
            ROS_WARN_STREAM("Waiting for initialization...");

            std::chrono::milliseconds dura(100);
            std::this_thread::sleep_for(dura);

            m_buf.unlock();
            continue;
        }
        else
        {
            input_img = ori_img_buf_.front();
            ori_img_buf_.pop();
        }
        m_buf.unlock();

        ObjectArray objects;
        yolox_->doInference(input_img, objects);
        cv::Mat output_img = vis_->drawObjects(input_img, objects);
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output_img).toImageMsg();
        pub_img.publish(msg);

        std::chrono::milliseconds dura(10);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "d457test_node");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if (argc != 2)
    {
        ROS_FATAL("rosrun realtime_detection d457test_node [config file].");
    }

    YoloxConfig config;
    std::string yaml_file = argv[1];
    readParameters(yaml_file, config);

    yolox_ = std::make_shared<YoloX>(config.input_model, config.resolution,
        config.score_threshold, config.nms_threshold, config.quantify);

    vis_ = std::make_shared<Visualization>();
    vis_->setLabelFile(config.label_path);

    ros::Subscriber sub_ori_img;
    sub_ori_img = n.subscribe("/camera/color/image_raw", 100, image_callback);

    image_transport::ImageTransport it(n);
    pub_img = it.advertise("/detection_res", 10);

    std::thread run_process{process};

    ros::spin();
    return 0;
}
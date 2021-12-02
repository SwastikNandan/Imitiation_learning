// Courtesy Harish Anand

#include "ros/ros.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "sensor_msgs/Image.h"
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <string>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int i = 0;
int a = 0;
int j = 0;
int b = 0;
image_transport::Publisher pub;

void depthToCV8UC1(const cv::Mat& float_img, cv::Mat& mono8_img){
    // Process images
    if(mono8_img.rows != float_img.rows || mono8_img.cols != float_img.cols){
        mono8_img = cv::Mat(float_img.size(), CV_8UC1);
    }
    cv::convertScaleAbs(float_img, mono8_img, 255.0/300.0, 0.0);
}

void depthCallback(const sensor_msgs::ImageConstPtr& original_image)
{
    cv_bridge::CvImagePtr cv_ptr;
    // Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    try
    {
        cv_ptr = cv_bridge::toCvCopy(original_image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
        return;
    }
    //Copy the image.data to imageBuf.
    cv::Mat depth_float_img = cv_ptr->image;
    cv::Mat depth_mono8_img;
    cv::patchNaNs(depth_float_img, 300.0);
    depthToCV8UC1(depth_float_img, depth_mono8_img);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", depth_mono8_img).toImageMsg();
    pub.publish(msg);
}

void rgbCallback(const sensor_msgs::ImageConstPtr& original_image)
{
    j++;
    if ( j % 90 != 0 ) {
        return;
    }
    cv_bridge::CvImagePtr cv_ptr;
    // Convert from the ROS image message to a CvImage suitable for working with OpenCV for processing
    try
    {
        cv_ptr = cv_bridge::toCvCopy(original_image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("tutorialROSOpenCV::main.cpp::cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat rgb_img = cv_ptr->image;
    cvtColor(rgb_img, rgb_img, CV_BGR2RGB);
    string path = "/home/swastik/iml_proj/rosbag/baylands/rgb/";
    string frame = "frame";
    string num = to_string(b);
    string new_num = std::string(4 - num.length(), '0') + num;
    string filename = path + frame + new_num + ".jpg";
    b = b + 1;
    imwrite(filename, rgb_img);
}

int main( int argc, char** argv ){
    ros::init(argc, argv, "talker");

    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber depth_sub = it.subscribe("/camera/depth/image_raw", 1, depthCallback);

    pub = it.advertise("/my_test/camera/image", 1);
    while (ros::ok())
    {
        ros::spinOnce();
    }

}



#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/point_cloud.h>
#include <pcl/range_image/range_image_planar.h>

#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/Pose.h>

namespace pcl_to_cv_proc{

void toCv(const pcl::RangeImagePlanar& range_image,
          cv::Mat& cv_image)
{

  cv_image = cv::Mat(range_image.height, range_image.width, CV_32FC1, cv::Scalar(std::numeric_limits<float>::max()));

  for (size_t i = 0; i < range_image.size(); ++i){
    // Invalid values in range image have -inf
    if (range_image[i].range > 0.0){
      cv_image.at<float>(i) = range_image[i].range;
    }
  }

}

template <typename PointT>
bool generateDepthImage(const pcl::PointCloud<PointT>& cloud,
                        const Eigen::Affine3d& camera_pose,
                        pcl::RangeImagePlanar& range_image,
                        cv::Mat& out)
{

  // We use camera info here as it allows for potentially
  // easier use with camera type sensors in the future
  sensor_msgs::CameraInfo camera_info;
  camera_info.width = 50;
  camera_info.height = camera_info.width;
  camera_info.K[0] = camera_info.width;
  camera_info.K[4] = camera_info.K[0];


  range_image.createFromPointCloudWithFixedSize( cloud,
                                                 static_cast<int>(camera_info.width),
                                                 static_cast<int>(camera_info.height),
                                                 static_cast<float>(camera_info.width*0.5),
                                                 static_cast<float>(camera_info.height*0.5f),
                                                 static_cast<float>(camera_info.K[0]),
                                                 static_cast<float>(camera_info.K[4]),
                                                 camera_pose.cast<float>(),
                                                 pcl::RangeImage::LASER_FRAME);

  toCv(range_image, out);

  //range_img.createFromPointCloudWithFixedSize
  return true;
}

template <typename PointT>
bool generateDepthImage(const sensor_msgs::PointCloud2& cloud,
                        const Eigen::Affine3d& camera_pose,
                        pcl::RangeImagePlanar& range_image,
                        cv::Mat& out)
{
  pcl::PointCloud<PointT> cloud_pcl;
  pcl::fromROSMsg(cloud, cloud_pcl);

  return generateDepthImage<PointT>(cloud_pcl,
                             camera_pose,
                             range_image,
                             out);
}



}

#endif

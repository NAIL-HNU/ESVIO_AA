#include <esvio_image_representation/ImageRepresentation.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <std_msgs/Float32.h>
#include <glog/logging.h>
#include <thread>
#include <cmath>
#include <vector>

// #define ESVIO_REPRESENTATION_LOG

namespace esvio_image_representation
{
  // Constructor for the ImageRepresentation class.
  ImageRepresentation::ImageRepresentation(ros::NodeHandle &nh, ros::NodeHandle nh_private) : nh_(nh)
  {
    // Set up subscribers for events, camera information, and synchronization signals.
    event_sub_ = nh_.subscribe("events", 0, &ImageRepresentation::eventsCallback, this);
    camera_info_sub_ = nh_.subscribe("camera_info", 1, &ImageRepresentation::cameraInfoCallback, this);
    sync_topic_ = nh_.subscribe("sync", 1, &ImageRepresentation::syncCallback, this);

    // Initialize the image transport and advertise topics for image representation.
    image_transport::ImageTransport it_(nh_);
    image_representation_pub_ = it_.advertise("image_representation", 5);
    image_representation_temp_pub_ = it_.advertise("image_representation_temp", 5);

    // Retrieve and set system parameters from the private node handle.
    nh_private.param<bool>("use_sim_time", bUse_Sim_Time_, true);
    int representation_mode;
    nh_private.param<int>("representation_mode", representation_mode, 0);
    representation_mode_ = static_cast<RepresentationMode>(representation_mode);
    nh_private.param<int>("median_blur_kernel_size", median_blur_kernel_size_, 1);
    nh_private.param<int>("max_event_queue_len", max_event_queue_length_, 20);
    std::cout << "\33[32m"
              << "representation_mode: " << representation_mode << "\33[0m" << std::endl;

    // Additional settings for image rectification and system initialization.
    nh_private.param<bool>("is_left", is_left_, true);
    bCamInfoAvailable_ = false;
    bSensorInitialized_ = false;
    if (pEventQueueMat_)
    {
      pEventQueueMat_->clear();
    }
    sensor_size_ = cv::Size(0, 0);

    // Load stereo camera usage and specific image processing parameters.
    nh_private.param<bool>("use_stereo_cam", bUseStereoCam_, true);
    nh_private.param<double>("decay_ms", decay_ms_, 30);
    decay_sec_ = decay_ms_ / 1000.0;
    nh_private.param<double>("rect_size", rect_size_, 80);
    nh_private.param<int>("TOS_k", k_tos_, 3);
    nh_private.param<int>("TOS_T", T_tos_, 241);
    nh_private.param<int>("substraction_delta", substraction_delta_, 5);
    SILC_bound_ = (2 * r_ + 1) * (2 * r_ + 1);
    time_aa_ = 36473.901006;

    // Advertise additional image topics if the mode is set to Linear_TS.
    if (representation_mode_ == Linear_TS)
    {
      dx_image_pub_ = it_.advertise("dx_image_pub_", 5);
      dy_image_pub_ = it_.advertise("dy_image_pub_", 5);
    }
  }

  ImageRepresentation::~ImageRepresentation()
  {
    image_representation_pub_.shutdown();
    image_representation_temp_pub_.shutdown();
  }

  void ImageRepresentation::init(int width, int height)
  {
    sensor_size_ = cv::Size(width, height);
    bSensorInitialized_ = true;
    pEventQueueMat_.reset(new EventQueueMat(width, height, max_event_queue_length_));
    ROS_INFO("Sensor size: (%d x %d)", sensor_size_.width, sensor_size_.height);

    representation_TS_ = cv::Mat::zeros(sensor_size_, CV_64F);
    representation_AA_ = cv::Mat::zeros(sensor_size_, CV_8UC1);
    representation_TOS_ = cv::Mat::zeros(sensor_size_, CV_8UC1);

    TS_temp_map = Eigen::MatrixXd::Constant(sensor_size_.height, sensor_size_.width, -10);
  }

  // Function to calculate the contrast of an image.
  double cal_contrast(cv::Mat &src_img)
  {
    double mean = 0;
    double sums = 0;

    // Calculate the mean of non-zero pixels.
    for (int y = 0; y < src_img.rows; y++)
    {
      for (int x = 0; x < src_img.cols; x++)
      {
        double pixel = src_img.ptr<double>(y)[x];
        mean += pixel;
        if (pixel != 0)
        {
          sums++;
        }
      }
    }

    // If there are no non-zero pixels, return 0 to avoid division by zero.
    if (sums == 0)
      return 0;

    mean = mean / sums;

    double contrast = 0;

    // Calculate the sum of squared differences from the mean for non-zero pixels.
    for (int y = 0; y < src_img.rows; y++)
    {
      for (int x = 0; x < src_img.cols; x++)
      {
        double pixel = src_img.ptr<double>(y)[x];
        if (pixel != 0)
        {
          contrast += pow((pixel - mean), 2);
        }
      }
    }

    // Normalize the contrast by the number of non-zero pixels.
    contrast = contrast / sums;
    return contrast;
  }

  void ImageRepresentation::createImageRepresentationAtTime(const ros::Time &external_sync_time)
  {
    // Lock the function to ensure thread safety while accessing shared resources.
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Return early if the sensor is not initialized or camera information is not available.
    if (!bSensorInitialized_ || !bCamInfoAvailable_)
      return;

    // Initialize matrices to hold the image data for different representation modes.
    cv::Mat filiter_image = cv::Mat::zeros(sensor_size_, CV_64F);
    cv::Mat rectangle_image = cv::Mat::zeros(cv::Size(80, 80), CV_8U);
    cv::Mat AA_copy = cv::Mat::zeros(sensor_size_, CV_8UC1);

    // Time Surface (TS) representation processing.
    if (representation_mode_ == TS)
    {
      cv::Mat divide_image = cv::Mat(sensor_size_, CV_64F, cv::Scalar(decay_sec_));
      representation_TS_.setTo(cv::Scalar(0));
      TicToc t;

      // Calculate the TS for each pixel in the sensor.
      for (int y = 0; y < sensor_size_.height; ++y)
      {
        for (int x = 0; x < sensor_size_.width; ++x)
        {
          dvs_msgs::Event most_recent_event_at_coordXY_before_T;
          if (pEventQueueMat_->getMostRecentEventBeforeT(x, y, external_sync_time, &most_recent_event_at_coordXY_before_T))
          {
            const ros::Time &most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
            if (most_recent_stamp_at_coordXY.toSec() > 0)
            {
              double dt = (external_sync_time - most_recent_stamp_at_coordXY).toSec();
              representation_TS_.at<double>(y, x) = -dt / decay_sec_;
            }
          }
          else
          {
            representation_TS_.at<double>(y, x) = -10;
          }
        }
      }

      // Apply exponential decay to the TS values to obtain the final representation.
      cv::exp(representation_TS_, representation_TS_);
      filiter_image = representation_TS_.clone();
      // std::cout << "use time:" << t.toc() << std::endl;
    }

    // Linear Time Surface (Linear_TS) processing for event-based cameras.
    if (representation_mode_ == Linear_TS)
    {
      double external_t = external_sync_time.toSec();
      representation_TS_.setTo(cv::Scalar(0));

      // Ensure there are events to process.
      if (t_.empty())
        return;

      x_buffer_.swap(x_);
      y_buffer_.swap(y_);
      t_buffer_.swap(t_);

      // Process each event and adjust the time surface map.
      cv::parallel_for_(cv::Range(0, t_buffer_.size()), [&](const cv::Range &range)
                        {
      for (int i = range.start; i < range.end; ++i)
      {
        TS_temp_map(y_buffer_[i], x_buffer_[i]) = t_buffer_[i] / decay_sec_;
      } });

      // Convert the eigen matrix to OpenCV, apply decay and normalize.
      cv::eigen2cv(TS_temp_map, representation_TS_);
      representation_TS_ -= external_t / decay_sec_;
      cv::exp(representation_TS_, representation_TS_);
    }

    // Angle of Arrival (AA) processing using contrast evaluation.
    if (representation_mode_ == AA)
    {
      cv::Mat sum_mat = cv::Mat::zeros(sensor_size_, CV_64F);
      cv::Mat sum_mat_2 = cv::Mat::zeros(sensor_size_, CV_64F);
      cv::Mat AA_map = cv::Mat::zeros(sensor_size_, CV_8UC1);

      int rect_size = 80;
      std::vector<cv::Rect> rect_rois(sensor_size_.height * sensor_size_.width / (rect_size * rect_size));

      // Create regions of interest based on the rect size.
      for (int x = 0; x < sensor_size_.width / rect_size; x++)
      {
        for (int y = 0; y < sensor_size_.height / rect_size; y++)
        {
          rect_rois[sensor_size_.width / rect_size * y + x] = cv::Rect(rect_size * x, rect_size * y, rect_size, rect_size);
        }
      }

      std::vector<bool> flags(sensor_size_.height * sensor_size_.width / (rect_size * rect_size));
      int all_flags = 0;
      for (int i = 0; i < sensor_size_.height * sensor_size_.width / (rect_size * rect_size); i++)
      {
        flags[i] = true;
      }

      auto it = InvolvedEvents_.begin(); // This is somehow different to the luvHarris paper. The paper may control the number of events used.
      ros::Time event_time = (*it).ts;
      int sums = 0;
      bool flag = true;

      // Process each event to compute the AA map.
      for (; it != InvolvedEvents_.end(); it++)
      {
        sums++;
        dvs_msgs::Event e = *it;
        if (sum_mat.at<double>(e.y, e.x) > 0)
        {
          AA_map.at<uchar>(e.y, e.x) = 255;
        }
        sum_mat.at<double>(e.y, e.x)++;
        if ((e.ts - event_time).toSec() >= 0.001)
        {
          cv::Mat sum_mat_copy = sum_mat.clone();
          for (int i = 0; i < sensor_size_.height * sensor_size_.width / (rect_size * rect_size); i++)
          {
            if (flags[i] == true)
            {

              cv::Mat rec_sum_mat = sum_mat_copy(rect_rois[i]).clone();
              double score = cal_contrast(rec_sum_mat);
              if (score > 0.5)
              {
                representation_AA_ = AA_map.clone();
                cv::Mat temp_image = representation_AA_(rect_rois[i]).clone();
                temp_image.copyTo(AA_copy(rect_rois[i]));
                rec_sum_mat.copyTo(sum_mat_2(rect_rois[i]));
                flags[i] = false;
                all_flags++;
              }
            }
          }
          event_time = e.ts;
        }
      }
      representation_AA_ = AA_map.clone();

      for (int i = 0; i < sensor_size_.height * sensor_size_.width / (rect_size * rect_size); i++)
      {
        if (flags[i] == true)
        {
          representation_AA_(rect_rois[i]).copyTo(AA_copy(rect_rois[i]));
          sum_mat(rect_rois[i]).clone().copyTo(sum_mat_2(rect_rois[i]));
          flags[i] = false;
        }
      }
      it--;
      sum_mat_2.convertTo(sum_mat_2, CV_8U);
      AA_copy = sum_mat_2.clone();
    }

    static cv_bridge::CvImage cv_image, cv_image_temp;
    static cv_bridge::CvImage dx_image, dy_image;
    cv_image.encoding = "mono8";
    cv_image_temp.encoding = "mono8";
    dx_image.encoding = sensor_msgs::image_encodings::TYPE_64FC1;
    dy_image.encoding = sensor_msgs::image_encodings::TYPE_64FC1;

    if (representation_mode_ == TS)
    {
      cv::Mat TS_img = cv::Mat::zeros(sensor_size_, CV_64F);
      TS_img = representation_TS_ * 255.0;
      TS_img.convertTo(TS_img, CV_8U);
      cv::medianBlur(TS_img, TS_img, 2 * median_blur_kernel_size_ + 1);
      cv_image.image = TS_img.clone();

      cv::Mat TS_img_temp = cv::Mat::zeros(sensor_size_, CV_64F);
      TS_img_temp = filiter_image * 255.0;
      TS_img_temp.convertTo(TS_img_temp, CV_8U);
      cv_image_temp.image = TS_img_temp.clone();
    }
    if (representation_mode_ == Linear_TS)
    {
      cv::Mat TS_img = cv::Mat::zeros(sensor_size_, CV_64F);

      TS_img = representation_TS_ * 255.0;
      TS_img.convertTo(TS_img, CV_8U);
      cv::medianBlur(TS_img, TS_img, 2 * median_blur_kernel_size_ + 1);
      cv::GaussianBlur(TS_img, cv_image_temp.image,
                       cv::Size(7, 7), 0.0);
      cv_image.image = TS_img.clone();
      cv::GaussianBlur(TS_img, TS_img,
                       cv::Size(5, 5), 0.0);
      cv::Mat const_img = cv::Mat::ones(sensor_size_, CV_8U);
      const_img = const_img * 255;
      if (bUseStereoCam_)
      {
        cv::remap(TS_img, TS_img, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
      }
      const_img = const_img - TS_img;
      cv_image_temp.image = const_img.clone();
      cv::Sobel(const_img, dx_image.image, CV_64FC1, 1, 0);
      cv::Sobel(const_img, dy_image.image, CV_64FC1, 0, 1);
    }

    if (representation_mode_ == AA)
    {
      cv_image.image = representation_AA_.clone();
      cv_image_temp.image = AA_copy.clone();
    }

    // Publishing the images based on the computed representation.
    if (bCamInfoAvailable_)
    {
      cv_bridge::CvImage cv_image2;
      cv_image2.encoding = cv_image.encoding;
      cv_image2.header.stamp = external_sync_time;
      cv_image2.image = cv::Mat::zeros(cv_image.image.size(), CV_8U);

      if (bUseStereoCam_)
      {
        cv::remap(cv_image.image, cv_image2.image, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
      }
      else
      {
        cv_image2.image = cv_image.image;
      }
      cv_bridge::CvImage cv_image2_temp;
      cv_image2_temp.encoding = cv_image_temp.encoding;
      cv_image2_temp.header.stamp = external_sync_time;
      cv_image2_temp.image = cv::Mat::zeros(cv_image.image.size(), CV_8U);
      if (bUseStereoCam_ && representation_mode_ != Linear_TS)
      {
        cv::remap(cv_image_temp.image, cv_image2_temp.image, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
      }
      else
      {
        cv_image2_temp.image = cv_image_temp.image;
      }
      image_representation_pub_.publish(cv_image2.toImageMsg());
      image_representation_temp_pub_.publish(cv_image2_temp.toImageMsg());

      if (representation_mode_ == Linear_TS)
      {
        dx_image.header.stamp = external_sync_time;
        dx_image_pub_.publish(dx_image.toImageMsg());
        dy_image.header.stamp = external_sync_time;
        dy_image_pub_.publish(dy_image.toImageMsg());
      }
    }
    if (representation_mode_ == Linear_TS)
    {
      x_.clear();
      y_.clear();
      t_.clear();
      InvolvedEvents_.clear();
    }
    else
    {
      InvolvedEvents_.clear();
    }
  }

  void ImageRepresentation::syncCallback(const std_msgs::TimeConstPtr &msg)
  {
    if (bUse_Sim_Time_)
      sync_time_ = ros::Time::now();
    else
      sync_time_ = msg->data;

    if (NUM_THREAD_REPRESENTATION == 1)
      createImageRepresentationAtTime(sync_time_);
  }

  void ImageRepresentation::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg)
  {
    if (bCamInfoAvailable_)
      return;

    cv::Size sensor_size(msg->width, msg->height);
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        camera_matrix_.at<double>(cv::Point(i, j)) = msg->K[i + j * 3];

    distortion_model_ = msg->distortion_model;
    dist_coeffs_ = cv::Mat(msg->D.size(), 1, CV_64F);
    for (int i = 0; i < msg->D.size(); i++)
      dist_coeffs_.at<double>(i) = msg->D[i];

    if (bUseStereoCam_)
    {
      rectification_matrix_ = cv::Mat(3, 3, CV_64F);
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          rectification_matrix_.at<double>(cv::Point(i, j)) = msg->R[i + j * 3];

      projection_matrix_ = cv::Mat(3, 4, CV_64F);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
          projection_matrix_.at<double>(cv::Point(i, j)) = msg->P[i + j * 4];

      if (distortion_model_ == "equidistant")
      {
        cv::fisheye::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                             rectification_matrix_, projection_matrix_,
                                             sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
        bCamInfoAvailable_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
      }
      else if (distortion_model_ == "plumb_bob")
      {
        cv::initUndistortRectifyMap(camera_matrix_, dist_coeffs_,
                                    rectification_matrix_, projection_matrix_,
                                    sensor_size, CV_32FC1, undistort_map1_, undistort_map2_);
        bCamInfoAvailable_ = true;
        ROS_INFO("Camera information is loaded (Distortion model %s).", distortion_model_.c_str());
      }
      else
      {
        ROS_ERROR_ONCE("Distortion model %s is not supported.", distortion_model_.c_str());
        bCamInfoAvailable_ = false;
        return;
      }

      /* pre-compute the undistorted-rectified look-up table */
      precomputed_rectified_points_ = Eigen::Matrix2Xd(2, sensor_size.height * sensor_size.width);
      // raw coordinates
      cv::Mat_<cv::Point2f> RawCoordinates(1, sensor_size.height * sensor_size.width);
      for (int y = 0; y < sensor_size.height; y++)
      {
        for (int x = 0; x < sensor_size.width; x++)
        {
          int index = y * sensor_size.width + x;
          RawCoordinates(index) = cv::Point2f((float)x, (float)y);
        }
      }
      // undistorted-rectified coordinates
      cv::Mat_<cv::Point2f> RectCoordinates(1, sensor_size.height * sensor_size.width);
      if (distortion_model_ == "plumb_bob")
      {
        cv::undistortPoints(RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
                            rectification_matrix_, projection_matrix_);
        ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
      }
      else if (distortion_model_ == "equidistant")
      {
        cv::fisheye::undistortPoints(
            RawCoordinates, RectCoordinates, camera_matrix_, dist_coeffs_,
            rectification_matrix_, projection_matrix_);
        ROS_INFO("Undistorted-Rectified Look-Up Table with Distortion model: %s", distortion_model_.c_str());
      }
      else
      {
        std::cout << "Unknown distortion model is provided." << std::endl;
        exit(-1);
      }
      // load look-up table
      for (size_t i = 0; i < sensor_size.height * sensor_size.width; i++)
      {
        precomputed_rectified_points_.col(i) = Eigen::Matrix<double, 2, 1>(
            RectCoordinates(i).x, RectCoordinates(i).y);
      }
      ROS_INFO("Undistorted-Rectified Look-Up Table has been computed.");
    }
    else
    {
      // TODO: calculate undistortion map
      bCamInfoAvailable_ = true;
    }
  }

  void ImageRepresentation::eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (!bSensorInitialized_)
      init(msg->width, msg->height);

    int skip = msg->events.size() / 7000, num = 0;

    for (const dvs_msgs::Event &e : msg->events)
    {

      if (representation_mode_ != AA)
      {
        if (skip > 0 && (num < skip))
        {
          num++;
          continue;
        }
        num = 0;
      }

      events_.push_back(e);

      if (representation_mode_ == Linear_TS)
      {
        x_.push_back(e.x);
        y_.push_back(e.y);
        t_.push_back(e.ts.toSec());
        InvolvedEvents_.push_back(e);
      }
      else
      {
        InvolvedEvents_.push_back(e);
      }

      int i = events_.size() - 2;
      while (i >= 0 && events_[i].ts > e.ts)
      {
        events_[i + 1] = events_[i];
        i--;
      }
      events_[i + 1] = e;

      const dvs_msgs::Event &last_event = events_.back();
      // pEventQueueMat_->insertEvent(last_event);

      last_time_ = e.ts;
    }
    clearEventQueue();
  }

  void ImageRepresentation::clearEventQueue()
  {
    static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 5000000;
    if (events_.size() > MAX_EVENT_QUEUE_LENGTH)
    {
      size_t remove_events = events_.size() - MAX_EVENT_QUEUE_LENGTH;
      events_.erase(events_.begin(), events_.begin() + remove_events);
    }
  }

} // namespace esvio_image_representation
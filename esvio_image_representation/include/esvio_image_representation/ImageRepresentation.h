#ifndef esvio_image_representation_H_
#define esvio_image_representation_H_

#include <ros/ros.h>
#include <std_msgs/Time.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <esvio_image_representation/TicToc.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <deque>
#include <mutex>
#include <Eigen/Eigen>
#include <vector>

namespace esvio_image_representation
{
#define NUM_THREAD_REPRESENTATION 1
  using EventQueue = std::deque<dvs_msgs::Event>;

  double cal_contrast(cv::Mat &src_img);
  void event_filter(cv::Mat &src_img);
  void edge_extraction(cv::Mat &src_img, cv::Mat &dst_img);

  class EventQueueMat
  {
  public:
    EventQueueMat(int width, int height, int queueLen)
    {
      width_ = width;
      height_ = height;
      queueLen_ = queueLen;
      eqMat_ = std::vector<EventQueue>(width_ * height_, EventQueue());
    }

    void insertEvent(const dvs_msgs::Event &e)
    {
      if (!insideImage(e.x, e.y))
        return;
      else
      {
        EventQueue &eq = getEventQueue(e.x, e.y);
        eq.push_back(e);
        while (eq.size() > queueLen_)
          eq.pop_front();
      }
    }

    bool getMostRecentEventBeforeT(
        const size_t x,
        const size_t y,
        const ros::Time &t,
        dvs_msgs::Event *ev)
    {
      if (!insideImage(x, y))
        return false;

      EventQueue &eq = getEventQueue(x, y);
      if (eq.empty())
        return false;

      for (auto it = eq.rbegin(); it != eq.rend(); ++it)
      {
        const dvs_msgs::Event &e = *it;
        if (e.ts < t)
        {
          *ev = *it;
          return true;
        }
      }
      return false;
    }

    void clear()
    {
      eqMat_.clear();
    }

    bool insideImage(const size_t x, const size_t y)
    {
      return !(x < 0 || x >= width_ || y < 0 || y >= height_);
    }

    inline EventQueue &getEventQueue(const size_t x, const size_t y)
    {
      return eqMat_[x + width_ * y];
    }

    size_t width_;
    size_t height_;
    size_t queueLen_;
    std::vector<EventQueue> eqMat_;
  };

  struct ROSTimeCmp
  {
    bool operator()(const ros::Time &a, const ros::Time &b) const
    {
      return a.toNSec() < b.toNSec();
    }
  };
  using GlobalEventQueue = std::map<ros::Time, dvs_msgs::Event, ROSTimeCmp>;

  inline static EventQueue::iterator EventBuffer_lower_bound(
      EventQueue &eb, ros::Time &t)
  {
    return std::lower_bound(eb.begin(), eb.end(), t,
                            [](const dvs_msgs::Event &e, const ros::Time &t)
                            { return e.ts.toSec() < t.toSec(); });
  }

  inline static EventQueue::iterator EventBuffer_upper_bound(
      EventQueue &eb, ros::Time &t)
  {
    return std::upper_bound(eb.begin(), eb.end(), t,
                            [](const ros::Time &t, const dvs_msgs::Event &e)
                            { return t.toSec() < e.ts.toSec(); });
  }

  class ImageRepresentation
  {
    // TODO:
    struct Job1
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      EventQueueMat *pEventQueueMat_;
      cv::Mat *pRepresentation_;
      size_t start_col_, end_col_;
      size_t start_row_, end_row_;
      size_t i_thread_;
      //    std::shared_ptr<std::vector<std::pair<double, std::pair<int, int> > > > ptr_ts_coord_vec_;
      std::vector<std::pair<double, std::pair<int, int>>> *ptr_ts_coord_vec_;
      ros::Time external_sync_time_;
      double TS_param_decay_sec_;
      int SILC2_param_r_;
      int DiST_para_alpha_;
      int Dist_para_rho_;
    };

    struct Job2
    {
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      EventQueue *pInvolvedEvents_;
      cv::Mat *pRepresentation_;
      size_t i_thread_;
      ros::Time external_sync_time_;
      int SILC_param_r_;
      int SILC_param_bound_;
      int TOS_param_k_;
      int TOS_param_T_;
    };

  public:
    ImageRepresentation(ros::NodeHandle &nh, ros::NodeHandle nh_private);
    virtual ~ImageRepresentation();

  private:
    ros::NodeHandle nh_;
    // core
    void init(int width, int height);
    // Support: TS, SI-TS, TOS, Dist
    void createImageRepresentationAtTime(const ros::Time &external_sync_time);

    // callbacks
    void syncCallback(const std_msgs::TimeConstPtr &msg);
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr &msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr &msg);

    // utils
    void clearEventQueue();
    void initStereoRectify();

    // calibration parameters
    cv::Mat camera_matrix_, dist_coeffs_;
    cv::Mat rectification_matrix_, projection_matrix_;
    std::string distortion_model_;
    cv::Mat undistort_map1_, undistort_map2_;
    Eigen::Matrix2Xd precomputed_rectified_points_;

    // sub & pub
    ros::Subscriber event_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Subscriber sync_topic_;
    image_transport::Publisher image_representation_pub_;

    image_transport::Publisher image_representation_temp_pub_;
    image_transport::Publisher dx_image_pub_, dy_image_pub_;

    // online parameters
    bool bCamInfoAvailable_;
    bool bUse_Sim_Time_;
    cv::Size sensor_size_;
    ros::Time sync_time_;
    bool bSensorInitialized_;

    // offline parameters TODO
    double decay_ms_;
    bool ignore_polarity_;
    int median_blur_kernel_size_;
    int max_event_queue_length_;
    int events_maintained_size_;
    double rect_size_;

    // containers
    EventQueue events_;
    EventQueue InvolvedEvents_;
    EventQueue InvolvedEvents0_;
    EventQueue InvolvedEvents1_;

    EventQueue InvolvedEvents_for_test;

    std::shared_ptr<EventQueueMat> pEventQueueMat_;
    std::shared_ptr<EventQueueMat> pEventQueueMat1_;
    std::shared_ptr<EventQueueMat> pEventQueueMat2_;
    GlobalEventQueue GlobalEventQueue_;
    double t_old_most_current_events_, t_new_most_current_events_;

    // for Linear_TS
    std::vector<size_t> x_buffer_;
    std::vector<size_t> y_buffer_;
    std::vector<double> t_buffer_;
    std::vector<size_t> x_;
    std::vector<size_t> y_;
    std::vector<double> t_;
    ros::Time last_time_;

    cv::Mat most_recent_ts_map_;
    cv::Mat representation_TS_;
    cv::Mat representation_AA_;
    cv::Mat representation_TOS_;

    Eigen::MatrixXd TS_temp_map;

    // for rectify
    cv::Mat undistmap1_, undistmap2_;
    bool is_left_;

    // thread mutex
    std::mutex data_mutex_;

    // Representation Mode
    // Time Surface: [1] HOTS: A Hierarchy of Event-Based Time-Surfaces for Pattern Recognition.
    // Speed-Invariant Time Surface: [2] Speed Invariant Time Surface for Learning to Detect Corner Points with Event-Based Cameras.
    // TOS: [3] luvHarris: A Practical Corner Detector for Event-cameras
    // Dist: [4] N-ImageNet: Towards Robust, Fine-Grained Object Recognition with Event Cameras
    enum RepresentationMode
    {
      TS,        // 0
      AA,        // 1
      Linear_TS, // 2
    } representation_mode_;

    // parameters
    bool bUseStereoCam_;
    double decay_sec_;   // TS param
    int r_, SILC_bound_; // SILC param
    int k_tos_, T_tos_;  // TOS param
    int alpha_, rho_;    // DiST param
    double truncate_;    // DiST param
    int substraction_delta_;
    double time_aa_;
    TicToc GAD_time;
  };
} // namespace esvio_image_representation
#endif // esvio_image_representation_H_
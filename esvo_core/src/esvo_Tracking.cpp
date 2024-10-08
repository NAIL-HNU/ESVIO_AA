#include <esvo_core/esvo_Tracking.h>
#include <esvo_core/tools/TicToc.h>
#include <esvo_core/tools/params_helper.h>
#include <minkindr_conversions/kindr_tf.h>
#include <tf/transform_broadcaster.h>
#include <sys/stat.h>

// #define ESVO_CORE_TRACKING_DEBUG
// #define ESVO_CORE_TRACKING_DEBUG

Eigen::Matrix3d Imu_R = Eigen::Matrix3d::Identity();
Eigen::Quaterniond Imu_q = Eigen::Quaterniond::Identity();
Eigen::Vector3d Imu_t = Eigen::Vector3d::Zero();

Eigen::Matrix4d Pose_T = Eigen::Matrix4d::Identity();

namespace esvo_core
{
  esvo_Tracking::esvo_Tracking(
      const ros::NodeHandle &nh,
      const ros::NodeHandle &nh_private) : nh_(nh),
                                           pnh_(nh_private),
                                           it_(nh),
                                           TS_left_sub_(nh_, "time_surface_left", 10),
                                           TS_right_sub_(nh_, "time_surface_right", 10),
                                           TS_sync_(ExactSyncPolicy(10), TS_left_sub_, TS_right_sub_),
                                           calibInfoDir_(tools::param(pnh_, "calibInfoDir", std::string(""))),
                                           camSysPtr_(new CameraSystem(calibInfoDir_, false)),
                                           rpConfigPtr_(new RegProblemConfig(
                                               tools::param(pnh_, "patch_size_X", 25),
                                               tools::param(pnh_, "patch_size_Y", 25),
                                               tools::param(pnh_, "kernelSize", 15),
                                               tools::param(pnh_, "LSnorm", std::string("l2")),
                                               tools::param(pnh_, "huber_threshold", 10.0),
                                               tools::param(pnh_, "invDepth_min_range", 0.0),
                                               tools::param(pnh_, "invDepth_max_range", 0.0),
                                               tools::param(pnh_, "MIN_NUM_EVENTS", 1000),
                                               tools::param(pnh_, "MAX_REGISTRATION_POINTS", 500),
                                               tools::param(pnh_, "BATCH_SIZE", 200),
                                               tools::param(pnh_, "MAX_ITERATION", 10))),
                                           rpType_((RegProblemType)((size_t)tools::param(pnh_, "RegProblemType", 0))),
                                           rpSolver_(camSysPtr_, rpConfigPtr_, rpType_, NUM_THREAD_TRACKING),
                                           ESVO_System_Status_("INITIALIZATION"),
                                           imu_data_(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()),
                                           ets_(IDLE)
  {
    // offline data
    dvs_frame_id_ = tools::param(pnh_, "dvs_frame_id", std::string("dvs"));
    world_frame_id_ = tools::param(pnh_, "world_frame_id", std::string("world"));

    /**** online parameters ***/
    tracking_rate_hz_ = tools::param(pnh_, "tracking_rate_hz", 100);
    TS_HISTORY_LENGTH_ = tools::param(pnh_, "TS_HISTORY_LENGTH", 100);
    REF_HISTORY_LENGTH_ = tools::param(pnh_, "REF_HISTORY_LENGTH", 5);
    bSaveTrajectory_ = tools::param(pnh_, "SAVE_TRAJECTORY", false);
    bVisualizeTrajectory_ = tools::param(pnh_, "VISUALIZE_TRAJECTORY", true);
    resultPath_ = tools::param(pnh_, "PATH_TO_SAVE_TRAJECTORY", std::string());
    nh_.setParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);

    Eigen::Vector3d ba(0, 0, 0);
    Eigen::Vector3d bg(0, 0, 0);

    imu_data_.initialization(ba, bg);

    // online data callbacks
    events_left_sub_ = nh_.subscribe<dvs_msgs::EventArray>(
        "events_left", 0, &esvo_Tracking::eventsCallback, this);
    TS_sync_.registerCallback(boost::bind(&esvo_Tracking::timeSurfaceCallback, this, _1, _2));
    tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/esvo_tracking/pose_pub", 1);
    path_pub_ = nh_.advertise<nav_msgs::Path>("/esvo_tracking/trajectory", 1);
    map_sub_ = nh_.subscribe("pointcloud", 0, &esvo_Tracking::refMapCallback, this);                // local map in the ref view.
    stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &esvo_Tracking::stampedPoseCallback, this); // for accessing the pose of the ref view.
    imu_sub_ = nh_.subscribe("/imu/data", 0, &esvo_Tracking::refImuCallback, this);                 // local map in the ref view.
    // imu_sub_ = nh_.subscribe("/davis/left/imu", 0, &esvo_Tracking::refImuCallback, this);// local map in the ref view.
    /*** For Visualization and Test ***/
    reprojMap_pub_left_ = it_.advertise("Reproj_Map_Left", 1);
    rpSolver_.setRegPublisher(&reprojMap_pub_left_);

    /*** Tracker ***/
    T_world_cur_ = Eigen::Matrix<double, 4, 4>::Identity();
    t_world_cur_ = last_t_world_cur_ = last_t_ = Eigen::Vector3d::Zero();
    std::thread TrackingThread(&esvo_Tracking::TrackingLoop, this);
    TrackingThread.detach();
  }

  esvo_Tracking::~esvo_Tracking()
  {
    pose_pub_.shutdown();
  }

  void esvo_Tracking::TrackingLoop()
  {
    ros::Rate r(tracking_rate_hz_);
    while (ros::ok())
    {
      // std::cout << __LINE__ <<"create new regProblem############################" << std::endl;
      // Keep Idling
      if (refPCMap_.size() < 1 || TS_history_.size() < 1)
      {
        // std::cout << refPCMap_.size() << "," << TS_history_.size() << std::endl;
        r.sleep();
        continue;
      }
      // std::cout << __LINE__ <<"create new regProblem############################" << std::endl;
      // Reset
      nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
      if (ESVO_System_Status_ == "INITIALIZATION" && ets_ == WORKING) // This is true when the system is reset from dynamic reconfigure
      {
        reset();
        r.sleep();
        continue;
      }
      // std::cout << __LINE__ <<"create new regProblem############################" << std::endl;
      if (ESVO_System_Status_ == "TERMINATE")
      {
        LOG(INFO) << "The tracking node is terminated manually...";
        break;
      }

      // Data Transfer (If mapping node had published refPC.)
      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (ref_.t_.toSec() < refPCMap_.rbegin()->first.toSec()) // new reference map arrived
          refDataTransferring();
        if (cur_.t_.toSec() < TS_history_.rbegin()->first.toSec()) // new observation arrived
        {
          if (ref_.t_.toSec() >= TS_history_.rbegin()->first.toSec())
          {
            LOG(INFO) << "The time_surface observation should be obtained after the reference frame";
            exit(-1);
          }
          if (!curDataTransferring())
          {
            continue;
          }
        }
        else
        {
          continue;
        }
      }
      // std::cout << "create new regProblem############################" << std::endl;
      // create new regProblem
      TicToc tt;
      double t_resetRegProblem, t_solve, t_pub_result, t_pub_gt;
#ifdef ESVO_CORE_TRACKING_DEBUG
      tt.tic();
#endif
      if (rpSolver_.resetRegProblem(&ref_, &cur_))
      {
#ifdef ESVO_CORE_TRACKING_DEBUG
        t_resetRegProblem = tt.toc();
        tt.tic();
#endif
        if (ets_ == IDLE)
          ets_ = WORKING;
        if (ESVO_System_Status_ != "WORKING")
        {
          nh_.setParam("/ESVO_SYSTEM_STATUS", "WORKING");
          std::cout << "WORKING############################" << std::endl;
        }

        if (rpType_ == REG_NUMERICAL)
          rpSolver_.solve_numerical();
        if (rpType_ == REG_ANALYTICAL)
          rpSolver_.solve_analytical();
#ifdef ESVO_CORE_TRACKING_DEBUG
        t_solve = tt.toc();
        tt.tic();
#endif
        T_world_cur_ = cur_.tr_.getTransformationMatrix();
        t_world_cur_ = T_world_cur_.block(0, 3, 3, 1);
        publishPose(cur_.t_, cur_.tr_);
        if (bVisualizeTrajectory_)
          publishPath(cur_.t_, cur_.tr_);
#ifdef ESVO_CORE_TRACKING_DEBUG
        t_pub_result = tt.toc();
#endif

        // save result and gt if available.
        if (bSaveTrajectory_)
        {
          // save results to listPose and listPoseGt
          lTimestamp_.push_back(std::to_string(cur_.t_.toSec()));
          lPose_.push_back(cur_.tr_.getTransformationMatrix());
        }
      }
      else
      {
        nh_.setParam("/ESVO_SYSTEM_STATUS", "INITIALIZATION");
        ets_ = IDLE;
        //      LOG(INFO) << "Tracking thread is IDLE";
      }

#ifdef ESVO_CORE_TRACKING_LOG
      double t_overall_count = 0;
      t_overall_count = t_resetRegProblem + t_solve + t_pub_result;
      LOG(INFO) << "\n";
      LOG(INFO) << "------------------------------------------------------------";
      LOG(INFO) << "--------------------Tracking Computation Cost---------------";
      LOG(INFO) << "------------------------------------------------------------";
      LOG(INFO) << "ResetRegProblem: " << t_resetRegProblem << " ms, (" << t_resetRegProblem / t_overall_count * 100 << "%).";
      LOG(INFO) << "Registration: " << t_solve << " ms, (" << t_solve / t_overall_count * 100 << "%).";
      LOG(INFO) << "pub result: " << t_pub_result << " ms, (" << t_pub_result / t_overall_count * 100 << "%).";
      LOG(INFO) << "Total Computation (" << rpSolver_.lmStatics_.nPoints_ << "): " << t_overall_count << " ms.";
      LOG(INFO) << "------------------------------------------------------------";
      LOG(INFO) << "------------------------------------------------------------";
#endif
      r.sleep();
    } // while
  }

  bool
  esvo_Tracking::refDataTransferring()
  {
    // load reference info
    ref_.t_ = ros::Time(refPCMap_.rbegin()->first.toSec());
    ros::Time t = ros::Time(refPCMap_.rbegin()->first.toSec() - 0.001);
    nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
    //  LOG(INFO) << "SYSTEM STATUS(T"
    if (ESVO_System_Status_ == "INITIALIZATION" && ets_ == IDLE)
      ref_.tr_.setIdentity();
    if (ESVO_System_Status_ == "WORKING" || (ESVO_System_Status_ == "INITIALIZATION" && ets_ == WORKING))
    {
      if (!getPoseAt(t, ref_.tr_, dvs_frame_id_))
      {
        LOG(INFO) << "ESVO_System_Status_: " << ESVO_System_Status_ << ", ref_.t_: " << ref_.t_.toNSec();
        LOG(INFO) << "Logic error ! There must be a pose for the given timestamp, because mapping has been finished.";
        exit(-1);
        return false;
      }
    }

    size_t numPoint = refPCMap_.rbegin()->second->size();
    ref_.vPointXYZPtr_.clear();
    ref_.vPointXYZPtr_.reserve(numPoint);
    auto PointXYZ_begin_it = refPCMap_.rbegin()->second->begin();
    auto PointXYZ_end_it = refPCMap_.rbegin()->second->end();
    while (PointXYZ_begin_it != PointXYZ_end_it)
    {
      ref_.vPointXYZPtr_.push_back(PointXYZ_begin_it.base()); // Copy the pointer of the pointXYZ
      PointXYZ_begin_it++;
    }
    return true;
  }

  bool
  esvo_Tracking::curDataTransferring()
  {
    // load current observation
    auto ev_last_it = EventBuffer_lower_bound(events_left_, cur_.t_);
    auto TS_it = TS_history_.rbegin();

    // TS_history may not be updated before the tracking loop excutes the data transfering
    if (cur_.t_ == TS_it->first)
      return false;
    cur_.t_ = TS_it->first;
    cur_.pTsObs_ = &TS_it->second;

    nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
    if (ESVO_System_Status_ == "INITIALIZATION" && ets_ == IDLE)
    {
      cur_.tr_ = ref_.tr_;
      //    LOG(INFO) << "(IDLE) Assign cur's ("<< cur_.t_.toNSec() << ") pose with ref's at " << ref_.t_.toNSec();
      // LOG(INFO) << " " << cur_.tr_.getTransformationMatrix() << " ";
    }
    if (ESVO_System_Status_ == "WORKING" || (ESVO_System_Status_ == "INITIALIZATION" && ets_ == WORKING))
    {
      curImuTransferring();
      // change imu
      // R_temp will be used to change the coordinate system from the IMU to the camera.
      Eigen::Matrix3d R_temp;
      R_temp << -1, 0, 0,
          0, -1, 0,
          0, 0, 1;
      Eigen::Matrix3d R_w_c = T_world_cur_.block(0, 0, 3, 3);

      Eigen::Quaterniond q1 = Eigen::Quaterniond::Identity();
      Eigen::Quaterniond q = q1.slerp(1, Imu_q);

      if (t_world_cur_ != Eigen::Vector3d::Zero() && last_t_world_cur_ != Eigen::Vector3d::Zero())
      {
        last_t_ = t_world_cur_ - last_t_world_cur_;
        last_t_world_cur_ = T_world_cur_.block(0, 3, 3, 1);
        T_world_cur_.block(0, 3, 3, 1) += last_t_ + R_temp * Imu_t;
      }
      else
      {
        last_t_world_cur_ = T_world_cur_.block(0, 3, 3, 1);
      }

      T_world_cur_.block(0, 0, 3, 3) = R_w_c * R_temp * q.toRotationMatrix() * R_temp.inverse();

      cur_.tr_ = Transformation(T_world_cur_);
      // LOG(INFO) << "(WORKING) Assign cur's ("<< cur_.t_.toNSec() << ") pose with T_world_cur.";
    }
    // Count the number of events occuring since the last observation.
    auto ev_cur_it = EventBuffer_lower_bound(events_left_, cur_.t_);
    cur_.numEventsSinceLastObs_ = std::distance(ev_last_it, ev_cur_it) + 1;
    return true;
  }

  bool esvo_Tracking::curImuTransferring()
  {
    // std::lock_guard<std::mutex> lock(imu_mutex_);
    auto TS_it = TS_history_.rbegin();
    double cur_TS_time = TS_it->first.toSec();
    double last_TS_time = (++TS_it)->first.toSec();
    // std::cout << "time of 2 TS:" << cur_TS_time - last_TS_time << std::endl;

    imu_data_.getPose(last_TS_time, cur_TS_time);
    // Eigen::Vector3d delta_p = imu_data_.delta_p;
    // Eigen::Quaterniond delta_q = imu_data_.delta_q;

    Imu_q = imu_data_.delta_q;
    Imu_R = Imu_q.toRotationMatrix();
    Imu_t = imu_data_.delta_p;
    // std::cout << "q:" << "\n"<< Imu_R << std::endl;
    return true;
  }

  void esvo_Tracking::reset()
  {
    // clear all maintained data
    ets_ = IDLE;
    TS_id_ = 0;
    TS_history_.clear();
    refPCMap_.clear();
    events_left_.clear();
  }

  /********************** Callback functions *****************************/
  void esvo_Tracking::refImuCallback(const sensor_msgs::ImuPtr &msg)
  {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    // std::cout << "Get Imu Data!!!!!!" << std::endl;
    // std::cout << imu_data_.linearized_ba << std::endl;
    Eigen::Vector3d acc, gyr;
    if (imu_data_.dt_buf.size() == 0)
    {

      acc[0] = msg->linear_acceleration.x;
      acc[1] = msg->linear_acceleration.y;
      acc[2] = msg->linear_acceleration.z;

      gyr[0] = msg->angular_velocity.x;
      gyr[1] = msg->angular_velocity.y;
      gyr[2] = msg->angular_velocity.z;
      if (imu_data_.last_time == 0)
      {
        imu_data_.begin_time = msg->header.stamp.toSec();
        imu_data_.push_back(0.001, acc, gyr);
        imu_data_.last_time = imu_data_.begin_time;
      }
      else
      {
        double dt = msg->header.stamp.toSec() - imu_data_.last_time;
        if (dt < 0)
          return;
        imu_data_.begin_time = msg->header.stamp.toSec();
        imu_data_.push_back(dt, acc, gyr);
      }
    }
    else
    {
      double time = msg->header.stamp.toSec();
      double dt = time - imu_data_.last_time;
      if (dt < 0)
        return;
      acc[0] = msg->linear_acceleration.x;
      acc[1] = msg->linear_acceleration.y;
      acc[2] = msg->linear_acceleration.z;

      gyr[0] = msg->angular_velocity.x;
      gyr[1] = msg->angular_velocity.y;
      gyr[2] = msg->angular_velocity.z;
      imu_data_.push_back(dt, acc, gyr);
      imu_data_.last_time = time;
      // std::cout << "delta_t:" << dt << std::endl;
    }
    // std::cout << "delta_q:" << imu_data_.delta_q.toRotationMatrix() << std::endl;
  }

  void esvo_Tracking::refMapCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    pcl::PCLPointCloud2 pcl_pc;
    pcl_conversions::toPCL(*msg, pcl_pc);
    PointCloud::Ptr PC_ptr(new PointCloud());
    pcl::fromPCLPointCloud2(pcl_pc, *PC_ptr);
    refPCMap_.emplace(msg->header.stamp, PC_ptr);
    while (refPCMap_.size() > REF_HISTORY_LENGTH_)
    {
      auto it = refPCMap_.begin();
      refPCMap_.erase(it);
    }
  }

  void esvo_Tracking::eventsCallback(
      const dvs_msgs::EventArray::ConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // add new ones and remove old ones
    for (const dvs_msgs::Event &e : msg->events)
    {
      events_left_.push_back(e);
      int i = events_left_.size() - 2;
      while (i >= 0 && events_left_[i].ts > e.ts) // we may have to sort the queue, just in case the raw event messages do not come in a chronological order.
      {
        events_left_[i + 1] = events_left_[i];
        i--;
      }
      events_left_[i + 1] = e;
    }
    clearEventQueue();
  }

  void esvo_Tracking::clearEventQueue()
  {
    static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 5000000;
    if (events_left_.size() > MAX_EVENT_QUEUE_LENGTH)
    {
      size_t remove_events = events_left_.size() - MAX_EVENT_QUEUE_LENGTH;
      events_left_.erase(events_left_.begin(), events_left_.begin() + remove_events);
    }
  }

  void esvo_Tracking::timeSurfaceCallback(
      const sensor_msgs::ImageConstPtr &time_surface_left,
      const sensor_msgs::ImageConstPtr &time_surface_right)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
    try
    {
      cv_ptr_left = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
      cv_ptr_right = cv_bridge::toCvCopy(time_surface_right, sensor_msgs::image_encodings::MONO8);
      cv::GaussianBlur(cv_ptr_left->image, cv_ptr_left->image, cv::Size(7, 7), 0.0);
      cv::GaussianBlur(cv_ptr_right->image, cv_ptr_right->image, cv::Size(7, 7), 0.0);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // push back the most current TS.
    ros::Time t_new_ts = time_surface_left->header.stamp;
    TS_history_.emplace(t_new_ts, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_, false));
    TS_id_++;

    // keep TS_history_'s size constant
    while (TS_history_.size() > TS_HISTORY_LENGTH_)
    {
      auto it = TS_history_.begin();
      TS_history_.erase(it);
    }
  }

  void esvo_Tracking::stampedPoseCallback(const geometry_msgs::PoseStampedConstPtr &msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // add pose to tf
    tf::Transform tf(
        tf::Quaternion(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w),
        tf::Vector3(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z));
    tf::StampedTransform st(tf, msg->header.stamp, msg->header.frame_id, dvs_frame_id_.c_str());
    tf_->setTransform(st);
    // broadcast the tf such that the nav_path messages can find the valid fixed frame "map".
    static tf::TransformBroadcaster br;
    br.sendTransform(st);
  }

  bool esvo_Tracking::getPoseAt(
      const ros::Time &t, esvo_core::Transformation &Tr, const std::string &source_frame)
  {
    std::string *err_msg = new std::string();
    if (!tf_->canTransform(world_frame_id_, source_frame, t, err_msg))
    {
      LOG(WARNING) << t.toNSec() << " : " << *err_msg;
      delete err_msg;
      return false;
    }
    else
    {
      tf::StampedTransform st;
      tf_->lookupTransform(world_frame_id_, source_frame, t, st);
      tf::transformTFToKindr(st, &Tr);
      return true;
    }
  }

  /************ publish results *******************/
  void esvo_Tracking::publishPose(const ros::Time &t, Transformation &tr)
  {
    geometry_msgs::PoseStampedPtr ps_ptr(new geometry_msgs::PoseStamped());

    ps_ptr->header.stamp = t;
    ps_ptr->header.frame_id = world_frame_id_;
    ps_ptr->pose.position.x = tr.getPosition()(0);
    ps_ptr->pose.position.y = tr.getPosition()(1);
    ps_ptr->pose.position.z = tr.getPosition()(2);
    ps_ptr->pose.orientation.x = tr.getRotation().x();
    ps_ptr->pose.orientation.y = tr.getRotation().y();
    ps_ptr->pose.orientation.z = tr.getRotation().z();
    ps_ptr->pose.orientation.w = tr.getRotation().w();
    pose_pub_.publish(ps_ptr);
  }

  void esvo_Tracking::publishPath(const ros::Time &t, Transformation &tr)
  {
    geometry_msgs::PoseStampedPtr ps_ptr(new geometry_msgs::PoseStamped());
    ps_ptr->header.stamp = t;
    ps_ptr->header.frame_id = world_frame_id_;
    ps_ptr->pose.position.x = tr.getPosition()(0);
    ps_ptr->pose.position.y = tr.getPosition()(1);
    ps_ptr->pose.position.z = tr.getPosition()(2);
    ps_ptr->pose.orientation.x = tr.getRotation().x();
    ps_ptr->pose.orientation.y = tr.getRotation().y();
    ps_ptr->pose.orientation.z = tr.getRotation().z();
    ps_ptr->pose.orientation.w = tr.getRotation().w();
    path_.header.stamp = t;
    path_.header.frame_id = world_frame_id_;
    path_.poses.push_back(*ps_ptr);
    path_pub_.publish(path_);

    if (!resultPath_.empty() && bSaveTrajectory_)
    {
      std::ofstream f;
      f.open(resultPath_ + "stamped_traj_estimate_ours.txt", std::ofstream::app);
      f << std::fixed;
      Eigen::Matrix3d Rwc_result;
      Eigen::Vector3d twc_result;
      f.setf(std::ios::fixed, std::ios::floatfield);
      f.precision(9);
      f << t << " ";
      f.precision(5);
      f << ps_ptr->pose.position.x << " "
        << ps_ptr->pose.position.y << " "
        << ps_ptr->pose.position.z << " "
        << ps_ptr->pose.orientation.x << " "
        << ps_ptr->pose.orientation.y << " "
        << ps_ptr->pose.orientation.z << " "
        << ps_ptr->pose.orientation.w << endl;
      f.close();
    }
  }

} // namespace esvo_core

#include <esvo_core/esvo2_Mapping.h>
#include <esvo_core/DVS_MappingStereoConfig.h>
#include <esvo_core/tools/params_helper.h>

#include <minkindr_conversions/kindr_tf.h>

#include <geometry_msgs/TransformStamped.h>

#include <opencv2/imgproc.hpp>
#include <cv_bridge/cv_bridge.h>

#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <thread>
#include <iterator>
#include <memory>
#include <algorithm>
#include <utility>
#include <map>
#include <iostream>

// #define ESVO_CORE_MAPPING_DEBUG
// #define ESVO_CORE_MAPPING_LOG

namespace esvo_core
{
  esvo2_Mapping::esvo2_Mapping(
      const ros::NodeHandle &nh,
      const ros::NodeHandle &nh_private)
      : nh_(nh),
        pnh_(nh_private),
        TS_left_sub_(nh_, "time_surface_left", 10),
        TS_right_sub_(nh_, "time_surface_right", 10),
        last_now_left_sub_(nh_, "last_now_left", 10),
        TS_sync_(ExactSyncPolicy(10), TS_left_sub_, TS_right_sub_),
        TS_TOS2_sync_(ExactSyncPolicy2(10), TS_left_sub_, TS_right_sub_, last_now_left_sub_),
        AA_left_sub_(nh_, "AA_left", 10),
        AA_right_sub_(nh_, "AA_right", 10),
        AA_sync_(ExactSyncPolicy(10), AA_left_sub_, AA_right_sub_),
        it_(nh),
        calibInfoDir_(tools::param(pnh_, "calibInfoDir", std::string(""))),
        camSysPtr_(new CameraSystem(calibInfoDir_, false)),
        dpConfigPtr_(new DepthProblemConfig(
            tools::param(pnh_, "patch_size_X", 25),
            tools::param(pnh_, "patch_size_Y", 25),
            tools::param(pnh_, "LSnorm", std::string("Tdist")),
            tools::param(pnh_, "Tdist_nu", 0.0),
            tools::param(pnh_, "Tdist_scale", 0.0),
            tools::param(pnh_, "ITERATION_OPTIMIZATION", 10),
            tools::param(pnh_, "RegularizationRadius", 5),
            tools::param(pnh_, "RegularizationMinNeighbours", 8),
            tools::param(pnh_, "RegularizationMinCloseNeighbours", 8))),
        dpSolver_(camSysPtr_, dpConfigPtr_, NUMERICAL, NUM_THREAD_MAPPING),
        dFusor_(camSysPtr_, dpConfigPtr_),
        dRegularizor_(dpConfigPtr_),
        ebm_(camSysPtr_, NUM_THREAD_MAPPING, tools::param(pnh_, "SmoothTimeSurface", false)),
        pc_(new PointCloud()),
        pc_near_(new PointCloud()),
        pc_global_(new PointCloud()),
        depthFramePtr_(new DepthFrame(camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_))
  {
    // frame id
    dvs_frame_id_ = tools::param(pnh_, "dvs_frame_id", std::string("dvs"));
    world_frame_id_ = tools::param(pnh_, "world_frame_id", std::string("world"));
    pc_->header.frame_id = world_frame_id_;
    pc_near_->header.frame_id = world_frame_id_;
    pc_global_->header.frame_id = world_frame_id_;

    /**** mapping parameters ***/
    // range and visualization threshold
    invDepth_min_range_ = tools::param(pnh_, "invDepth_min_range", 0.16);
    invDepth_max_range_ = tools::param(pnh_, "invDepth_max_range", 2.0);
    patch_area_ = tools::param(pnh_, "patch_size_X", 25) * tools::param(pnh_, "patch_size_Y", 25);
    residual_vis_threshold_ = tools::param(pnh_, "residual_vis_threshold", 15);
    cost_vis_threshold_ = pow(residual_vis_threshold_, 2) * patch_area_;
    stdVar_vis_threshold_ = tools::param(pnh_, "stdVar_vis_threshold", 0.005);
    age_max_range_ = tools::param(pnh_, "age_max_range", 5);
    age_vis_threshold_ = tools::param(pnh_, "age_vis_threshold", 0);
    fusion_radius_ = tools::param(pnh_, "fusion_radius", 0);
    maxNumFusionFrames_ = tools::param(pnh_, "maxNumFusionFrames", 10);
    FusionStrategy_ = tools::param(pnh_, "FUSION_STRATEGY", std::string("CONST_FRAMES"));
    maxNumFusionPoints_ = tools::param(pnh_, "maxNumFusionPoints", 2000);
    INIT_SGM_DP_NUM_Threshold_ = tools::param(pnh_, "INIT_SGM_DP_NUM_THRESHOLD", 500);
    // options
    bDenoising_ = tools::param(pnh_, "Denoising", false);
    bRegularization_ = tools::param(pnh_, "Regularization", false);
    resetButton_ = tools::param(pnh_, "ResetButton", false);
    // visualization parameters
    bVisualizeGlobalPC_ = tools::param(pnh_, "bVisualizeGlobalPC", false);
    visualizeGPC_interval_ = tools::param(pnh_, "visualizeGPC_interval", 3);
    visualize_range_ = tools::param(pnh_, "visualize_range", 2.5);
    numAddedPC_threshold_ = tools::param(pnh_, "NumGPC_added_per_refresh", 1000);
    // module parameters
    PROCESS_EVENT_NUM_ = tools::param(pnh_, "PROCESS_EVENT_NUM", 500);
    PROCESS_EVENT_NUM_AA_ = tools::param(pnh_, "PROCESS_EVENT_NUM_AA", 500);
    TS_HISTORY_LENGTH_ = tools::param(pnh_, "TS_HISTORY_LENGTH", 100);
    USE_AA_for_BM_ = tools::param(pnh_, "USE_AA_for_BM", false);
    mapping_rate_hz_ = tools::param(pnh_, "mapping_rate_hz", 20);
    // Event Block Matching (BM) parameters
    BM_half_slice_thickness_ = tools::param(pnh_, "BM_half_slice_thickness", 0.001);
    BM_patch_size_X_ = tools::param(pnh_, "patch_size_X", 25);
    BM_patch_size_Y_ = tools::param(pnh_, "patch_size_Y", 25);
    BM_min_disparity_ = tools::param(pnh_, "BM_min_disparity", 3);
    BM_max_disparity_ = tools::param(pnh_, "BM_max_disparity", 40);
    BM_step_ = tools::param(pnh_, "BM_step", 1);
    BM_ZNCC_Threshold_ = tools::param(pnh_, "BM_ZNCC_Threshold", 0.1);
    BM_bUpDownConfiguration_ = tools::param(pnh_, "BM_bUpDownConfiguration", false);

    // SGM parameters (Used by Initialization)
    num_disparities_ = 16 * 3;
    block_size_ = 11;
    P1_ = 8 * 1 * block_size_ * block_size_;
    P2_ = 32 * 1 * block_size_ * block_size_;
    uniqueness_ratio_ = 11;
    sgbm_ = cv::StereoSGBM::create(0, num_disparities_, block_size_, P1_, P2_,
                                   -1, 0, uniqueness_ratio_);

    // calcualte the min,max disparity
    double f = (camSysPtr_->cam_left_ptr_->P_(0, 0) + camSysPtr_->cam_left_ptr_->P_(1, 1)) / 2;
    double b = camSysPtr_->baseline_;
    size_t minDisparity = max(size_t(std::floor(f * b * invDepth_min_range_)), (size_t)0);
    size_t maxDisparity = size_t(std::ceil(f * b * invDepth_max_range_));
    minDisparity = max(minDisparity, BM_min_disparity_);
    maxDisparity = min(maxDisparity, BM_max_disparity_);

#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << "f: " << f << " "
              << " b: " << b;
    LOG(INFO) << "invDepth_min_range_: " << invDepth_min_range_;
    LOG(INFO) << "invDepth_max_range_: " << invDepth_max_range_;
    LOG(INFO) << "minDisparity: " << minDisparity;
    LOG(INFO) << "maxDisparity: " << maxDisparity;
#endif

    // initialize Event Batch Matcher
    ebm_.resetParameters(BM_patch_size_X_, BM_patch_size_Y_, minDisparity, maxDisparity,
                         BM_step_, BM_ZNCC_Threshold_, BM_bUpDownConfiguration_);

    // system status
    ESVO_System_Status_ = "INITIALIZATION";
    nh_.setParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);

    // callback functions
    // events_left_sub_  = nh_.subscribe<dvs_msgs::EventArray>("events_left", 0, boost::bind(&esvo2_Mapping::eventsCallback, this, _1, boost::ref(events_left_)));
    // events_right_sub_ = nh_.subscribe<dvs_msgs::EventArray>("events_right", 0, boost::bind(&esvo2_Mapping::eventsCallback, this, _1, boost::ref(events_right_)));
    stampedPose_sub_ = nh_.subscribe("stamped_pose", 0, &esvo2_Mapping::stampedPoseCallback, this);
    if (USE_AA_for_BM_)
      TS_TOS2_sync_.registerCallback(boost::bind(&esvo2_Mapping::timeSurfaceCallback2, this, _1, _2, _3));
    else
      TS_sync_.registerCallback(boost::bind(&esvo2_Mapping::timeSurfaceCallback, this, _1, _2));
    AA_sync_.registerCallback(boost::bind(&esvo2_Mapping::AACallback, this, _1, _2, boost::ref(events_left_)));
    // TF
    tf_ = std::make_shared<tf::Transformer>(true, ros::Duration(100.0));

    // result publishers
    invDepthMap_pub_ = it_.advertise("Inverse_Depth_Map2", 1);
    invDepthMap_rel_pub_ = it_.advertise("Inverse_Depth_rel_Map2", 1);
    stdVarMap_pub_ = it_.advertise("Standard_Variance_Map2", 1);
    ageMap_pub_ = it_.advertise("Age_Map2", 1);
    costMap_pub_ = it_.advertise("cost_map2", 1);
    pc_pub_ = nh_.advertise<PointCloud>("/esvo2_mapping/pointcloud_local2", 1);
    if (bVisualizeGlobalPC_)
    {
      gpc_pub_ = nh_.advertise<PointCloud>("/esvo2_mapping/pointcloud_global2", 1);
      pc_global_->reserve(5000000);
      t_last_pub_pc_ = 0.0;
    }

    // multi-thread management
    mapping_thread_future_ = mapping_thread_promise_.get_future();
    reset_future_ = reset_promise_.get_future();

    // stereo mapping detached thread
    std::thread MappingThread(&esvo2_Mapping::MappingLoop, this,
                              std::move(mapping_thread_promise_), std::move(reset_future_));
    MappingThread.detach();

    // Dynamic reconfigure
    dynamic_reconfigure_callback_ = boost::bind(&esvo2_Mapping::onlineParameterChangeCallback, this, _1, _2);
    server_.reset(new dynamic_reconfigure::Server<DVS_MappingStereoConfig>(nh_private));
    server_->setCallback(dynamic_reconfigure_callback_);
  }

  esvo2_Mapping::~esvo2_Mapping()
  {
    pc_pub_.shutdown();
    invDepthMap_pub_.shutdown();
    invDepthMap_rel_pub_.shutdown();
    stdVarMap_pub_.shutdown();
    ageMap_pub_.shutdown();
    costMap_pub_.shutdown();
  }

  void esvo2_Mapping::MappingLoop(
      std::promise<void> prom_mapping,
      std::future<void> future_reset)
  {
    ros::Rate r(mapping_rate_hz_);

    while (ros::ok())
    {
      // reset mapping rate
      if (changed_frame_rate_)
      {
        r = ros::Rate(mapping_rate_hz_);
        changed_frame_rate_ = false;
      }
      // check system status
      nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
      //    LOG(INFO) << "SYSTEM STATUS (MappingLoop): " << ESVO_System_Status_;
      if (ESVO_System_Status_ == "TERMINATE")
      {
        LOG(INFO) << "The Mapping node is terminated manually...";
        break;
      }
      if (TS_history_.size() >= 10) /* To assure the esvo_time_surface node has been working. */
      {
        while (true)
        {
          if (data_mutex_.try_lock())
          {
            dataTransferring();
            data_mutex_.unlock();
            break;
          }
          else
          {
            if (future_reset.wait_for(std::chrono::nanoseconds(1)) == std::future_status::ready)
            {
              prom_mapping.set_value();
              return;
            }
          }
        }
        // To check if the most current TS observation has been loaded by dataTransferring()
        if (TS_obs_.second.isEmpty())
        {
          r.sleep();
          continue;
        }
        // Do initialization (State Machine)
        if (ESVO_System_Status_ == "INITIALIZATION" || ESVO_System_Status_ == "RESET")
        {
#ifdef ESVO_CORE_MAPPING_DEBUG
          TicToc tt;
          tt.tic();
#endif
          if (InitializationAtTime(TS_obs_.first))
          {
            LOG(INFO) << "Initialization is successfully done!";
#ifdef ESVO_CORE_MAPPING_DEBUG
            LOG(INFO) << "Initialization takes: " << tt.toc() << " ms.";
#endif
          }
          else
            LOG(INFO) << "Initialization fails once.";
        }
        // Do mapping
        if (ESVO_System_Status_ == "WORKING")
          MappingAtTime(TS_obs_.first);
      }
      else
      {
        if (future_reset.wait_for(std::chrono::nanoseconds(1)) == std::future_status::ready)
        {
          prom_mapping.set_value();
          return;
        }
      }
      r.sleep();
    }
  }

  void esvo2_Mapping::MappingAtTime(const ros::Time &t)
  {
    TicToc tt_mapping;
    double t_overall_count = 0;
    /************************************************/
    /************ set the new DepthFrame ************/
    /************************************************/
    DepthFrame::Ptr depthFramePtr_new = std::make_shared<DepthFrame>(
        camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);
    depthFramePtr_new->setId(TS_obs_.second.id_);
    depthFramePtr_new->setTransformation(TS_obs_.second.tr_);
    depthFramePtr_ = depthFramePtr_new;

    std::vector<EventMatchPair> vEMP; // the container that stores the result of BM.
    /****************************************************/
    /*************** Block Matching (BM) ****************/
    /****************************************************/
    double t_BM = 0.0;
    double t_BM_denoising = 0.0;

    // Denoising operations
    if (bDenoising_) // Set it to "True" to deal with flicker effect caused by VICON.
    {
      tt_mapping.tic();
      // Draw one mask image for denoising.
      cv::Mat denoising_mask;
      createDenoisingMask(vALLEventsPtr_left_, denoising_mask,
                          camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);

      // Extract denoised events (appear on edges likely).
      vDenoisedEventsPtr_left_.clear();
      extractDenoisedEvents(vCloseEventsPtr_left_, vDenoisedEventsPtr_left_, denoising_mask, PROCESS_EVENT_NUM_);
      totalNumCount_ = vDenoisedEventsPtr_left_.size();

      t_BM_denoising = tt_mapping.toc();
    }
    else
    {
      vDenoisedEventsPtr_left_.clear();
      vDenoisedEventsPtr_left_.reserve(PROCESS_EVENT_NUM_);
      vDenoisedEventsPtr_left_.insert(
          vDenoisedEventsPtr_left_.end(), vCloseEventsPtr_left_.begin(),
          vCloseEventsPtr_left_.begin() + min(vCloseEventsPtr_left_.size(), PROCESS_EVENT_NUM_));
    }

    // block matching
    tt_mapping.tic();
    cv::Mat tmp_TS_l, tmp_TS_r, tmp_TS_last, ori, stereo_result;

    // draw stereo_result
    cv::Mat dx_map, dy_map;
    cv::Mat tmp_TS_l2;
    tmp_TS_l = TS_obs_.second.cvImagePtr_left_->image.clone();
    tmp_TS_r = TS_obs_.second.cvImagePtr_right_->image.clone();

    // compute dx_dy
    tmp_TS_l.convertTo(tmp_TS_l2, CV_64F);
    tmp_TS_l2 = tmp_TS_l2 / 200;
    Scharr(tmp_TS_l2, TS_obs_.second.x_grad_, CV_64F, 1, 0, 3); // compute gradient dx
    Scharr(tmp_TS_l2, TS_obs_.second.y_grad_, CV_64F, 0, 1, 3); // compute gradient dy
    cv::convertScaleAbs(TS_obs_.second.x_grad_, TS_obs_.second.x_grad_);
    cv::convertScaleAbs(TS_obs_.second.y_grad_, TS_obs_.second.y_grad_);
    cv::normalize(TS_obs_.second.x_grad_, TS_obs_.second.x_grad_, 0, 1, cv::NORM_MINMAX);
    cv::normalize(TS_obs_.second.y_grad_, TS_obs_.second.y_grad_, 0, 1, cv::NORM_MINMAX);
    TS_obs_.second.x_grad_ = TS_obs_.second.x_grad_ * 255;
    TS_obs_.second.y_grad_ = TS_obs_.second.y_grad_ * 255;
    cv::GaussianBlur(TS_obs_.second.x_grad_, TS_obs_.second.x_grad_, cv::Size(9, 9), 0);
    cv::GaussianBlur(TS_obs_.second.y_grad_, TS_obs_.second.y_grad_, cv::Size(9, 9), 0);
    TS_obs_.second.x_grad_.convertTo(dx_map, CV_8U);
    TS_obs_.second.y_grad_.convertTo(dy_map, CV_8U);

    // gaussian blur
    cv::Mat TS_gaussian;
    TS_gaussian = cv::Mat::zeros(TS_obs_.second.cvImagePtr_left_->image.size(), CV_8U);
    TS_gaussian = TS_obs_.second.cvImagePtr_left_->image.clone();
    cv::GaussianBlur(TS_gaussian, TS_gaussian, cv::Size(7, 7), 0);
    TS_gaussian.convertTo(TS_gaussian, CV_8U);

    // separate events by dx_dy and gaussian blur
    vDenoisedEventsPtr_left_dx_.clear();
    vDenoisedEventsPtr_left_dy_.clear();
    vDenoisedEventsPtr_left_dx_.reserve(PROCESS_EVENT_NUM_);
    vDenoisedEventsPtr_left_dy_.reserve(PROCESS_EVENT_NUM_);

    cv::GaussianBlur(dx_map, dx_map, cv::Size(9, 9), 0);
    cv::GaussianBlur(dx_map, dx_map, cv::Size(9, 9), 0);

    for (auto event : vDenoisedEventsPtr_left_)
    {
      if (event->y - 4 < 0 || event->y + 4 >= dx_map.rows - 1 || event->x - 4 < 0 || event->x + 4 >= dx_map.cols - 1 || TS_gaussian.at<uchar>(event->y, event->x) < 5)
      {
        continue;
      }

      double dx = dx_map.at<uchar>(event->y, event->x);
      double dy = dy_map.at<uchar>(event->y, event->x);

      // Adjust dx and dy if they are zero to avoid division by zero
      dy = (dy == 0) ? 0.1 : dy;
      dx = (dx == 0) ? 0.1 : dx;

      // Compute the ratio of dx to dy
      double dx_dy = dx / dy;

      // Classify the event based on the ratio of dx to dy
      if (dx_dy < 0.2)
      {
        vDenoisedEventsPtr_left_dy_.push_back(event);
      }
      else if (dx_dy > 0.2)
      {
        vDenoisedEventsPtr_left_dx_.push_back(event);
      }
    }

    // Create a matching problem with observed data, state map, and filtered events by gradient dx.
    ebm_.createMatchProblem(&TS_obs_, &st_map_, &vDenoisedEventsPtr_left_dx_);
    ebm_.match_all_HyperThread(vEMP);
#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << ">>>> The number of input events: " << vDenoisedEventsPtr_left_.size();
    LOG(INFO) << "++++ Block Matching (BM) generates " << vEMP.size() << " candidates.";
#endif

    std::vector<EventMatchPair> vEMP_last_pre, vEMP_last, vEMP_last_fail;
    vEMP_last_pre.reserve(vDenoisedEventsPtr_left_dy_.size());
    vEMP_last.reserve(vDenoisedEventsPtr_left_dy_.size());
    vEMP_last_fail.reserve(vDenoisedEventsPtr_left_dy_.size());
    if (!TS_obs_.second.isLastEmpty())
    {
      // T_last_now = T_w_last^(-1) * T_w_now
      Eigen::Matrix4d T_last_now = TS_obs_.second.tr_last_.getTransformationMatrix().inverse() * TS_obs_.second.tr_.getTransformationMatrix();
      getReprojection(vEMP_last_pre, T_last_now, vDenoisedEventsPtr_left_dy_);
      tmp_TS_l = cv::Mat::zeros(TS_obs_.second.cvImagePtr_left_->image.size(), CV_8U);
      tmp_TS_last = cv::Mat::zeros(TS_obs_.second.cvImagePtr_left_->image.size(), CV_8U);

      // choose map to do block matching: AA or TS
      if (USE_AA_for_BM_)
      {
        tmp_TS_l = TS_obs_.second.cvImagePtr_last_now_left_->image.clone();
        tmp_TS_last = TS_obs_.second.cvImagePtr_last_->image.clone();
      }
      else
      {
        tmp_TS_l = TS_obs_.second.cvImagePtr_left_->image.clone();
        tmp_TS_last = TS_obs_.second.cvImagePtr_last_->image.clone();
      }
      cv::cv2eigen(tmp_TS_l, TS_obs_.second.TS_left_);
      cv::cv2eigen(tmp_TS_last, TS_obs_.second.TS_last_);

      // Create a matching problem for two frames using observed data, state map, events filtered by gradient dy, and previous frame data.
      ebm_.createMatchProblemTwoFrames(&TS_obs_, &st_map_, &vDenoisedEventsPtr_left_dy_, &vEMP_last_pre);
      ebm_.match_all_HyperThreadTwoFrames(vEMP_last, vEMP_last_fail);
      cv::cv2eigen(TS_obs_.second.cvImagePtr_left_->image, TS_obs_.second.TS_left_);
      std::cout << "num of match between left and right: " << vEMP.size() << "====num of match between last and now: " << vEMP_last.size() << " =======before ln: " << vEMP_last_pre.size() << std::endl;
      vEMP.insert(vEMP.end(), vEMP_last.begin(), vEMP_last.begin() + vEMP_last.size());
      std::cout << "====num of all match: " << vEMP.size() << std::endl;
    }

    t_BM = tt_mapping.toc();
    t_overall_count += t_BM_denoising;
    t_overall_count += t_BM;

    /**************************************************************/
    /*************  Nonlinear Optimization & Fusion ***************/
    /**************************************************************/
    double t_optimization = 0;
    double t_solve, t_fusion, t_regularization;
    t_solve = t_fusion = t_regularization = 0;
    size_t numFusionCount = 0; // To count the total number of fusion (in terms of fusion between two estimates, i.e. a priori and a propagated one).

    tt_mapping.tic();
    // nonlinear opitmization
    std::vector<DepthPoint> vdp;
    vdp.reserve(vEMP.size());
    dpSolver_.solve(&vEMP, &TS_obs_, vdp); // hyper-thread version
    dpSolver_.pointCulling(vdp, stdVar_vis_threshold_, cost_vis_threshold_,
                           invDepth_min_range_, invDepth_max_range_);
    t_solve = tt_mapping.toc();

    // Fusion (strategy 1: const number of point)
    if (FusionStrategy_ == "CONST_POINTS")
    {
      size_t numFusionPoints = 0;
      tt_mapping.tic();
      dqvDepthPoints_.push_back(vdp);
      for (size_t n = 0; n < dqvDepthPoints_.size(); n++)
        numFusionPoints += dqvDepthPoints_[n].size();
      while (numFusionPoints > 1.5 * maxNumFusionPoints_)
      {
        dqvDepthPoints_.pop_front();
        numFusionPoints = 0;
        for (size_t n = 0; n < dqvDepthPoints_.size(); n++)
          numFusionPoints += dqvDepthPoints_[n].size();
      }
    }
    else if (FusionStrategy_ == "CONST_FRAMES") // (strategy 2: const number of frames)
    {
      tt_mapping.tic();
      dqvDepthPoints_.push_back(vdp);
      while (dqvDepthPoints_.size() > maxNumFusionFrames_)
        dqvDepthPoints_.pop_front();
    }
    else
    {
      LOG(INFO) << "Invalid FusionStrategy is assigned.";
      exit(-1);
    }

    // apply fusion and count the total number of fusion.
    numFusionCount = 0;
    for (auto it = dqvDepthPoints_.rbegin(); it != dqvDepthPoints_.rend(); it++)
    {
      numFusionCount += dFusor_.update(*it, depthFramePtr_, fusion_radius_);
    }

    TotalNumFusion_ += numFusionCount;
    // Now let's play gently. We do not clean the resulting depth map at the beginning (as long as the number of "frames"
    // to be fused is less than maxNumFusionFrames_), because the depth estimates given by SGM (initialization) would
    // not be fused if the T-distribution statistics is used. Therefore, to guarantee a stable tracking at the beginning,
    // we have to preserve as much depth points as possible. Note that, typically, most depth estimates got cleaned because
    // of their "age", which is an attribute reflecting the "goodness of fit" besides standard variance.
    if (dqvDepthPoints_.size() >= maxNumFusionFrames_)
      depthFramePtr_->dMap_->clean(pow(stdVar_vis_threshold_, 2), age_vis_threshold_, invDepth_max_range_, invDepth_min_range_);
    t_fusion = tt_mapping.toc();

    // regularization
    if (bRegularization_)
    {
      tt_mapping.tic();
      dRegularizor_.apply(depthFramePtr_->dMap_);
      t_regularization = tt_mapping.toc();
    }
    // count time
    t_optimization = t_solve + t_fusion + t_regularization;
    t_overall_count += t_optimization;

    // publish results
    std::thread tPublishMappingResult(&esvo2_Mapping::publishMappingResults, this,
                                      depthFramePtr_->dMap_, depthFramePtr_->T_world_frame_, t);
    tPublishMappingResult.detach();

    LOG(INFO) << "\n";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "--------------Computation Cost (Mapping)---------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "Denoising: " << t_BM_denoising << " ms, (" << t_BM_denoising / t_overall_count * 100 << "%).";
    LOG(INFO) << "Block Matching (BM): " << t_BM << " ms, (" << t_BM / t_overall_count * 100 << "%).";
    LOG(INFO) << "BM success ratio: " << vEMP.size() << "/" << totalNumCount_ << "(Successes/Total).";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "Update: " << t_optimization << " ms, (" << t_optimization / t_overall_count * 100
              << "%).";
    LOG(INFO) << "-- nonlinear optimization: " << t_solve << " ms, (" << t_solve / t_overall_count * 100
              << "%).";
    LOG(INFO) << "-- fusion (" << numFusionCount << ", " << TotalNumFusion_ << "): " << t_fusion << " ms, (" << t_fusion / t_overall_count * 100
              << "%).";
    LOG(INFO) << "-- regularization: " << t_regularization << " ms, (" << t_regularization / t_overall_count * 100
              << "%).";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "Total Computation (" << depthFramePtr_->dMap_->size() << "): " << t_overall_count << " ms.";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "------------------------------END---------------------------";
    LOG(INFO) << "------------------------------------------------------------";
    LOG(INFO) << "\n";
  }

  bool esvo2_Mapping::InitializationAtTime(const ros::Time &t)
  {
    // create a new depth frame
    DepthFrame::Ptr depthFramePtr_new = std::make_shared<DepthFrame>(
        camSysPtr_->cam_left_ptr_->height_, camSysPtr_->cam_left_ptr_->width_);
    depthFramePtr_new->setId(TS_obs_.second.id_);
    depthFramePtr_new->setTransformation(TS_obs_.second.tr_);
    depthFramePtr_ = depthFramePtr_new;

    // call SGM on the current Time Surface observation pair.
    cv::Mat dispMap, dispMap8;
    sgbm_->compute(TS_obs_.second.cvImagePtr_left_->image, TS_obs_.second.cvImagePtr_right_->image, dispMap);
    dispMap.convertTo(dispMap8, CV_8U, 255 / (num_disparities_ * 16.));

    // get the event map (binary mask)
    cv::Mat edgeMap;
    std::vector<std::pair<size_t, size_t>> vEdgeletCoordinates;
    createEdgeMask(vEventsPtr_left_SGM_, camSysPtr_->cam_left_ptr_,
                   edgeMap, vEdgeletCoordinates, true, 0);

    // Apply logical "AND" operation and transfer "disparity" to "invDepth".
    std::vector<DepthPoint> vdp_sgm;
    vdp_sgm.reserve(vEdgeletCoordinates.size());
    double var_SGM = pow(0.001, 2);
    for (size_t i = 0; i < vEdgeletCoordinates.size(); i++)
    {
      size_t x = vEdgeletCoordinates[i].first;
      size_t y = vEdgeletCoordinates[i].second;

      double disp = dispMap.at<short>(y, x) / 16.0;
      if (disp < 0)
        continue;
      DepthPoint dp(x, y);
      Eigen::Vector2d p_img(x * 1.0, y * 1.0);
      dp.update_x(p_img);
      double invDepth = disp / (camSysPtr_->cam_left_ptr_->P_(0, 0) * camSysPtr_->baseline_);
      if (invDepth < invDepth_min_range_ || invDepth > invDepth_max_range_)
        continue;
      Eigen::Vector3d p_cam;
      camSysPtr_->cam_left_ptr_->cam2World(p_img, invDepth, p_cam);
      dp.update_p_cam(p_cam);
      dp.update(invDepth, var_SGM); // assume the statics of the SGM's results are Guassian.
      dp.residual() = 0.0;
      dp.age() = age_vis_threshold_;
      Eigen::Matrix<double, 4, 4> T_world_cam = TS_obs_.second.tr_.getTransformationMatrix();
      dp.updatePose(T_world_cam);
      vdp_sgm.push_back(dp);
    }
    LOG(INFO) << "********** Initialization (SGM) returns " << vdp_sgm.size() << " points.";
    if (vdp_sgm.size() < INIT_SGM_DP_NUM_Threshold_)
      return false;
    // push the "masked" SGM results to the depthFrame
    dqvDepthPoints_.push_back(vdp_sgm);
    dFusor_.naive_propagation(vdp_sgm, depthFramePtr_);
    // publish the invDepth map
    std::thread tPublishMappingResult(&esvo2_Mapping::publishMappingResults, this,
                                      depthFramePtr_->dMap_, depthFramePtr_->T_world_frame_, t);
    tPublishMappingResult.detach();
    return true;
  }

  bool esvo2_Mapping::dataTransferring()
  {
    TS_obs_ = std::make_pair(ros::Time(), TimeSurfaceObservation()); // clean the TS obs.
    if (TS_history_.size() <= 10)                                    /* To assure the esvo_time_surface node has been working. */
      return false;
    totalNumCount_ = 0;

    // load current Time-Surface Observation
    auto it_end = TS_history_.rbegin();
    it_end++; // in case that the tf is behind the most current TS.
    auto it_begin = TS_history_.begin();
    while (TS_obs_.second.isEmpty())
    {
      Transformation tr, tr_last;
      if (ESVO_System_Status_ == "INITIALIZATION")
      {
        tr.setIdentity();
        it_end->second.setTransformation(tr);
        TS_obs_ = *it_end;
      }
      if (ESVO_System_Status_ == "WORKING")
      {
        if (getPoseAt(it_end->first, tr, dvs_frame_id_))
        {
          it_end->second.setTransformation(tr);
          TS_obs_ = *it_end;
          while (it_end->first != it_begin->first)
          {
            if (getPoseAt(it_end->first, tr_last, dvs_frame_id_))
            {

              if ((tr_last.getPosition() - tr.getPosition()).norm() > 0.3)
              {
                it_end->second.setTransformation(tr_last);
                Last_obs_ = *it_end;
                if (!Last_obs_.second.isEmpty())
                {
                  TS_obs_.second.tr_last_ = Last_obs_.second.tr_;
                  if (USE_AA_for_BM_)
                  {
                    TS_obs_.second.TS_last_ = Last_obs_.second.last_now_left_;
                    TS_obs_.second.cvImagePtr_last_ = Last_obs_.second.cvImagePtr_last_now_left_;
                  }
                  else
                  {
                    TS_obs_.second.TS_last_ = Last_obs_.second.TS_left_;
                    TS_obs_.second.cvImagePtr_last_ = Last_obs_.second.cvImagePtr_left_;
                  }
                }

                break;
              }
            }
            it_end++;
          }
        }
        else
        {
          // check if the tracking node is still working normally
          nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
          if (ESVO_System_Status_ != "WORKING")
            return false;
        }
      }
      if (it_end->first == it_begin->first)
        break;
      it_end++;
    }
    if (TS_obs_.second.isEmpty())
      return false;

    /****** Load involved events *****/
    // SGM
    if (ESVO_System_Status_ == "INITIALIZATION")
    {
      vEventsPtr_left_SGM_.clear();
      ros::Time t_end = TS_obs_.first;
      ros::Time t_begin(std::max(0.0, t_end.toSec() - 2 * BM_half_slice_thickness_ - 0.04));
      // ros::Time t_begin(std::max(0.0, t_end.toSec() - 2 * BM_half_slice_thickness_));
      auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end);
      auto ev_begin_it = tools::EventBuffer_lower_bound(events_left_, t_begin);
      const size_t MAX_NUM_Event_INVOLVED = 30000;
      vEventsPtr_left_SGM_.reserve(MAX_NUM_Event_INVOLVED);
      while (ev_begin_it != ev_end_it && vEventsPtr_left_SGM_.size() <= PROCESS_EVENT_NUM_)
      {
        vEventsPtr_left_SGM_.push_back(ev_begin_it._M_cur);
        ev_begin_it++;
      }
    }

    // BM
    if (ESVO_System_Status_ == "WORKING")
    {
      // copy all involved events' pointers
      vALLEventsPtr_left_.clear();   // Used to generate denoising mask (only used to deal with flicker induced by VICON.)
      vCloseEventsPtr_left_.clear(); // Will be denoised using the mask above.

      // load allEvent
      ros::Time t_end = TS_obs_.first;
      ros::Time t_begin(std::max(0.0, t_end.toSec() - 0.035));
      auto ev_end_it = tools::EventBuffer_lower_bound(events_left_, t_end);
      auto ev_begin_it = tools::EventBuffer_lower_bound(events_left_, t_begin);
      const size_t MAX_NUM_Event_INVOLVED = PROCESS_EVENT_NUM_;
      vALLEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
      vCloseEventsPtr_left_.reserve(MAX_NUM_Event_INVOLVED);
      while (ev_end_it != ev_begin_it && vALLEventsPtr_left_.size() < MAX_NUM_Event_INVOLVED)
      {
        vALLEventsPtr_left_.push_back(ev_end_it._M_cur);
        vCloseEventsPtr_left_.push_back(ev_end_it._M_cur);
        ev_end_it--;
      }
      totalNumCount_ = vCloseEventsPtr_left_.size();
      // #ifdef ESVO_CORE_MAPPING_DEBUG
      LOG(INFO) << "Data Transferring (events_left_): " << events_left_.size();
      LOG(INFO) << "Data Transferring (vALLEventsPtr_left_): " << vALLEventsPtr_left_.size();
      LOG(INFO) << "Data Transforming (vCloseEventsPtr_left_): " << vCloseEventsPtr_left_.size();
      // #endif

      // Ideally, each event occurs at an unique perspective (virtual view) -- pose.
      // In practice, this is intractable in real-time application.
      // We made a trade off by assuming that events occurred within (0.05 * BM_half_slice_thickness_) ms share an identical pose (virtual view).
      // Here we load transformations for all virtual views.
      st_map_.clear();
      ros::Time t_tmp = t_begin;
      while (t_tmp.toSec() <= t_end.toSec())
      {
        Transformation tr;
        if (getPoseAt(t_tmp, tr, dvs_frame_id_))
          st_map_.emplace(t_tmp, tr);
        else
        {
          nh_.getParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
          if (ESVO_System_Status_ != "WORKING")
            return false;
        }
        t_tmp = ros::Time(t_tmp.toSec() + 0.05 * BM_half_slice_thickness_);
      }
#ifdef ESVO_CORE_MAPPING_DEBUG
      LOG(INFO) << "Data Transferring (stampTransformation map): " << st_map_.size();
#endif
    }
    return true;
  }

  void esvo2_Mapping::stampedPoseCallback(
      const geometry_msgs::PoseStampedConstPtr &ps_msg)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // To check inconsistent timestamps and reset.
    static constexpr double max_time_diff_before_reset_s = 0.5;
    const ros::Time stamp_first_event = ps_msg->header.stamp;
    std::string *err_tf = new std::string();
    //  int iGetLastest_common_time =
    //    tf_->getLatestCommonTime(dvs_frame_id_.c_str(), ps_msg->header.frame_id, tf_lastest_common_time_, err_tf);
    delete err_tf;

    if (tf_lastest_common_time_.toSec() != 0)
    {
      const double dt = stamp_first_event.toSec() - tf_lastest_common_time_.toSec();
      if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
      {
        ROS_INFO("Inconsistent event timestamps detected <stampedPoseCallback> (new: %f, old %f), resetting.",
                 stamp_first_event.toSec(), tf_lastest_common_time_.toSec());
        reset();
      }
    }

    // add pose to tf
    tf::Transform tf(
        tf::Quaternion(
            ps_msg->pose.orientation.x,
            ps_msg->pose.orientation.y,
            ps_msg->pose.orientation.z,
            ps_msg->pose.orientation.w),
        tf::Vector3(
            ps_msg->pose.position.x,
            ps_msg->pose.position.y,
            ps_msg->pose.position.z));
    tf::StampedTransform st(tf, ps_msg->header.stamp, ps_msg->header.frame_id, dvs_frame_id_.c_str());
    tf_->setTransform(st);
  }

  // return the pose of the left event cam at time t.
  bool esvo2_Mapping::getPoseAt(
      const ros::Time &t,
      esvo_core::Transformation &Tr, // T_world_virtual
      const std::string &source_frame)
  {
    std::string *err_msg = new std::string();
    if (!tf_->canTransform(world_frame_id_, source_frame, t, err_msg))
    {
#ifdef ESVO_CORE_MAPPING_LOG
      LOG(WARNING) << t.toNSec() << " : " << *err_msg;
#endif
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

  void esvo2_Mapping::eventsCallback(
      const dvs_msgs::EventArray::ConstPtr &msg,
      EventQueue &EQ)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);

    static constexpr double max_time_diff_before_reset_s = 0.5;
    const ros::Time stamp_first_event = msg->events[0].ts;

    // check time stamp inconsistency
    if (!msg->events.empty() && !EQ.empty())
    {
      const double dt = stamp_first_event.toSec() - EQ.back().ts.toSec();
      if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
      {
        ROS_INFO("Inconsistent event timestamps detected <eventCallback> (new: %f, old %f), resetting.",
                 stamp_first_event.toSec(), events_left_.back().ts.toSec());
        reset();
      }
    }

    // add new ones and remove old ones
    for (const dvs_msgs::Event &e : msg->events)
    {
      EQ.push_back(e);
      int i = EQ.size() - 2;
      while (i >= 0 && EQ[i].ts > e.ts) // we may have to sort the queue, just in case the raw event messages do not come in a chronological order.
      {
        EQ[i + 1] = EQ[i];
        i--;
      }
      EQ[i + 1] = e;
    }
    clearEventQueue(EQ);
  }

  void esvo2_Mapping::clearEventQueue(EventQueue &EQ)
  {
    static constexpr size_t MAX_EVENT_QUEUE_LENGTH = 3000000;
    if (EQ.size() > MAX_EVENT_QUEUE_LENGTH)
    {
      size_t NUM_EVENTS_TO_REMOVE = EQ.size() - MAX_EVENT_QUEUE_LENGTH;
      EQ.erase(EQ.begin(), EQ.begin() + NUM_EVENTS_TO_REMOVE);
    }
  }

  void esvo2_Mapping::timeSurfaceCallback(
      const sensor_msgs::ImageConstPtr &time_surface_left,
      const sensor_msgs::ImageConstPtr &time_surface_right)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // check time-stamp inconsistency
    if (!TS_history_.empty())
    {
      static constexpr double max_time_diff_before_reset_s = 1;
      const ros::Time stamp_last_image = TS_history_.rbegin()->first;
      const double dt = time_surface_left->header.stamp.toSec() - stamp_last_image.toSec();
      if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
      {
        ROS_INFO("Inconsistent frame timestamp detected <timeSurfaceCallback> (new: %f, old %f), resetting.",
                 time_surface_left->header.stamp.toSec(), stamp_last_image.toSec());
        reset();
      }
    }

    cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
    try
    {
      cv_ptr_left = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
      cv_ptr_right = cv_bridge::toCvCopy(time_surface_right, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // push back the new time surface map
    ros::Time t_new_TS = time_surface_left->header.stamp;
    // Made the gradient computation optional which is up to the jacobian choice.
    if (dpSolver_.getProblemType() == NUMERICAL)
      TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_));
    else
      TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, TS_id_, true));
    TS_id_++;

    // keep TS_history's size constant
    while (TS_history_.size() > TS_HISTORY_LENGTH_)
    {
      auto it = TS_history_.begin();
      TS_history_.erase(it);
    }
  }

  void esvo2_Mapping::timeSurfaceCallback2(
      const sensor_msgs::ImageConstPtr &time_surface_left,
      const sensor_msgs::ImageConstPtr &time_surface_right,
      const sensor_msgs::ImageConstPtr &last_now_left)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // check time-stamp inconsistency
    if (!TS_history_.empty())
    {
      static constexpr double max_time_diff_before_reset_s = 1;
      const ros::Time stamp_last_image = TS_history_.rbegin()->first;
      const double dt = time_surface_left->header.stamp.toSec() - stamp_last_image.toSec();
      if (dt < 0 || std::fabs(dt) >= max_time_diff_before_reset_s)
      {
        ROS_INFO("Inconsistent frame timestamp detected <timeSurfaceCallback> (new: %f, old %f), resetting.",
                 time_surface_left->header.stamp.toSec(), stamp_last_image.toSec());
        reset();
      }
    }

    cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right, cv_ptr_last_now_left;
    try
    {
      cv_ptr_left = cv_bridge::toCvCopy(time_surface_left, sensor_msgs::image_encodings::MONO8);
      cv_ptr_right = cv_bridge::toCvCopy(time_surface_right, sensor_msgs::image_encodings::MONO8);
      cv_ptr_last_now_left = cv_bridge::toCvCopy(last_now_left, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // push back the new time surface map
    ros::Time t_new_TS = time_surface_left->header.stamp;
    // Made the gradient computation optional which is up to the jacobian choice.
    if (dpSolver_.getProblemType() == NUMERICAL)
      TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, cv_ptr_last_now_left, TS_id_));
    else
      TS_history_.emplace(t_new_TS, TimeSurfaceObservation(cv_ptr_left, cv_ptr_right, cv_ptr_last_now_left, TS_id_, true));
    TS_id_++;

    // keep TS_history's size constant
    while (TS_history_.size() > TS_HISTORY_LENGTH_)
    {
      auto it = TS_history_.begin();
      TS_history_.erase(it);
    }
  }

  void esvo2_Mapping::AACallback(
      const sensor_msgs::ImageConstPtr &AA_left,
      const sensor_msgs::ImageConstPtr &AA_right,
      EventQueue &EQ)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // check time-stamp inconsistency
    if (!TS_history_.empty())
    {
      static constexpr double max_time_diff_before_reset_s = 1;
      const ros::Time stamp_last_image = TS_history_.rbegin()->first;
      const double dt = AA_left->header.stamp.toSec() - stamp_last_image.toSec();
      if (std::fabs(dt) >= max_time_diff_before_reset_s)
      {
        ROS_INFO("Inconsistent frame timestamp detected <AACallback> (new: %f, old %f), resetting.",
                 AA_left->header.stamp.toSec(), stamp_last_image.toSec());
        reset();
      }
    }

    cv_bridge::CvImagePtr cv_ptr_left, cv_ptr_right;
    try
    {
      cv_ptr_left = cv_bridge::toCvCopy(AA_left, sensor_msgs::image_encodings::MONO8);
      // cv_ptr_right = cv_bridge::toCvCopy(AA_right, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    // select the pixels with high events rate
    int num_of_resultImg = 0, drift_t = 0;
    double persent_of_point = 1;
    EventQueue EQ_tmp;
    cv::Mat resultImg = cv_ptr_left->image.clone();
    int rect_size = 80;
    std::vector<std::vector<std::pair<int, cv::Point>>> roi_events(resultImg.rows * resultImg.cols / (rect_size * rect_size));
    std::vector<int> num_of_roi(resultImg.rows * resultImg.cols / (rect_size * rect_size), 0);
    cv::Mat AA = cv::Mat::zeros(resultImg.size(), resultImg.type());
    for (int y = 0; y < resultImg.rows; y++)
    {
      for (int x = 0; x < resultImg.cols; x++)
      {
        if (resultImg.at<uchar>(y, x) > 0)
        {
          num_of_resultImg++;
          int index = (y / rect_size) * (resultImg.cols / rect_size) + (x / rect_size);
          num_of_roi[index]++;
          roi_events[index].push_back(std::make_pair((int)resultImg.at<uchar>(y, x), cv::Point(x, y)));
        }
      }
    }
    if (num_of_resultImg > PROCESS_EVENT_NUM_AA_)
    {
      persent_of_point = ((double)PROCESS_EVENT_NUM_AA_) / num_of_resultImg;
    }
    for (int i = 0; i < (resultImg.rows * resultImg.cols / (rect_size * rect_size)); i++)
    {
      sort(roi_events[i].begin(), roi_events[i].end(), [](std::pair<int, cv::Point> a, std::pair<int, cv::Point> b)
           { return (a.first > b.first); });
      for (int j = 0; j < roi_events[i].size() * persent_of_point; j++)
      {
        dvs_msgs::Event e;

        e.x = roi_events[i][j].second.x;
        e.y = roi_events[i][j].second.y;
        e.ts = ros::Time(AA_left->header.stamp.toSec() - drift_t * 1e-11);
        EQ_tmp.push_back(e);
        drift_t++;
        AA.at<uchar>(e.y, e.x) = 255;
      }
    }
    AA_now_ = AA.clone();

    // add new ones and remove old ones
    for (const dvs_msgs::Event &e : EQ_tmp)
    {
      EQ.push_back(e);
      int i = EQ.size() - 2;
      while (i >= 0 && EQ[i].ts > e.ts) // we may have to sort the queue, just in case the raw event messages do not come in a chronological order.
      {
        EQ[i + 1] = EQ[i];
        i--;
      }
      EQ[i + 1] = e;
    }
    clearEventQueue(EQ);
  }

  void esvo2_Mapping::reset()
  {
    // mutual-thread communication with MappingThread.
    LOG(INFO) << "Coming into reset()";
    reset_promise_.set_value();
    LOG(INFO) << "(reset) The mapping thread future is waiting for the value.";
    mapping_thread_future_.get();
    LOG(INFO) << "(reset) The mapping thread future receives the value.";

    // clear all maintained data
    events_left_.clear();
    events_right_.clear();
    TS_history_.clear();
    tf_->clear();
    pc_->clear();
    pc_near_->clear();
    pc_global_->clear();
    TS_id_ = 0;
    depthFramePtr_->clear();
    dqvDepthPoints_.clear();

    ebm_.resetParameters(BM_patch_size_X_, BM_patch_size_Y_,
                         BM_min_disparity_, BM_max_disparity_, BM_step_, BM_ZNCC_Threshold_, BM_bUpDownConfiguration_);

    for (int i = 0; i < 2; i++)
      LOG(INFO) << "****************************************************";
    LOG(INFO) << "****************** RESET THE SYSTEM *********************";
    for (int i = 0; i < 2; i++)
      LOG(INFO) << "****************************************************\n\n";

    // restart the mapping thread
    reset_promise_ = std::promise<void>();
    mapping_thread_promise_ = std::promise<void>();
    reset_future_ = reset_promise_.get_future();
    mapping_thread_future_ = mapping_thread_promise_.get_future();
    ESVO_System_Status_ = "INITIALIZATION";
    nh_.setParam("/ESVO_SYSTEM_STATUS", ESVO_System_Status_);
    std::thread MappingThread(&esvo2_Mapping::MappingLoop, this,
                              std::move(mapping_thread_promise_), std::move(reset_future_));
    MappingThread.detach();
  }

  void esvo2_Mapping::onlineParameterChangeCallback(DVS_MappingStereoConfig &config, uint32_t level)
  {
    bool online_parameters_changed = false;
    {
      std::lock_guard<std::mutex> lock(data_mutex_);

      if (invDepth_min_range_ != config.invDepth_min_range ||
          invDepth_max_range_ != config.invDepth_max_range ||
          residual_vis_threshold_ != config.residual_vis_threshold ||
          stdVar_vis_threshold_ != config.stdVar_vis_threshold ||
          age_max_range_ != config.age_max_range ||
          age_vis_threshold_ != config.age_vis_threshold ||
          fusion_radius_ != config.fusion_radius ||
          maxNumFusionFrames_ != config.maxNumFusionFrames ||
          bDenoising_ != config.Denoising ||
          bRegularization_ != config.Regularization ||
          resetButton_ != config.ResetButton ||
          PROCESS_EVENT_NUM_ != config.PROCESS_EVENT_NUM ||
          TS_HISTORY_LENGTH_ != config.TS_HISTORY_LENGTH ||
          BM_min_disparity_ != config.BM_min_disparity ||
          BM_max_disparity_ != config.BM_max_disparity ||
          BM_step_ != config.BM_step ||
          BM_ZNCC_Threshold_ != config.BM_ZNCC_Threshold)
      {
        online_parameters_changed = true;
      }

      invDepth_min_range_ = config.invDepth_min_range;
      invDepth_max_range_ = config.invDepth_max_range;
      residual_vis_threshold_ = config.residual_vis_threshold;
      cost_vis_threshold_ = patch_area_ * pow(residual_vis_threshold_, 2);
      stdVar_vis_threshold_ = config.stdVar_vis_threshold;
      age_max_range_ = config.age_max_range;
      age_vis_threshold_ = config.age_vis_threshold;
      fusion_radius_ = config.fusion_radius;
      maxNumFusionFrames_ = config.maxNumFusionFrames;
      bDenoising_ = config.Denoising;
      bRegularization_ = config.Regularization;
      resetButton_ = config.ResetButton;
      PROCESS_EVENT_NUM_ = config.PROCESS_EVENT_NUM;
      TS_HISTORY_LENGTH_ = config.TS_HISTORY_LENGTH;
      BM_min_disparity_ = config.BM_min_disparity;
      BM_max_disparity_ = config.BM_max_disparity;
      BM_step_ = config.BM_step;
      BM_ZNCC_Threshold_ = config.BM_ZNCC_Threshold;
    }

    if (config.mapping_rate_hz != mapping_rate_hz_)
    {
      changed_frame_rate_ = true;
      online_parameters_changed = true;
      mapping_rate_hz_ = config.mapping_rate_hz;
    }

    if (online_parameters_changed)
    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      LOG(INFO) << "onlineParameterChangeCallback ==============";
      reset();
    }
  }

  void esvo2_Mapping::publishMappingResults(
      DepthMap::Ptr depthMapPtr,
      Transformation tr,
      ros::Time t)
  {
    cv::Mat invDepthImage, stdVarImage, ageImage, costImage, eventImage, confidenceMap, invDepthImage_rel;

    visualizor_.plot_map(depthMapPtr, tools::InvDepthMap, invDepthImage,
                         invDepth_max_range_, invDepth_min_range_, stdVar_vis_threshold_, age_vis_threshold_);
    publishImage(invDepthImage, t, invDepthMap_pub_);

    visualizor_.plot_map(depthMapPtr, tools::InvDepthMap_rel2, invDepthImage_rel,
                         invDepth_max_range_, invDepth_min_range_, stdVar_vis_threshold_, age_vis_threshold_);
    // publishImage(invDepthImage_rel, t, invDepthMap_rel_pub_);

    visualizor_.plot_map(depthMapPtr, tools::StdVarMap, stdVarImage,
                         stdVar_vis_threshold_, 0.0, stdVar_vis_threshold_);
    publishImage(stdVarImage, t, stdVarMap_pub_);

    visualizor_.plot_map(depthMapPtr, tools::AgeMap, ageImage, age_max_range_, 0, age_vis_threshold_);
    publishImage(ageImage, t, ageMap_pub_);

    visualizor_.plot_map(depthMapPtr, tools::CostMap, costImage, cost_vis_threshold_, 0.0, cost_vis_threshold_);
    publishImage(costImage, t, costMap_pub_);

    if (ESVO_System_Status_ == "INITIALIZATION")
      publishPointCloud(depthMapPtr, tr, t);
    if (ESVO_System_Status_ == "WORKING")
    {
      if (FusionStrategy_ == "CONST_FRAMES")
      {
        if (dqvDepthPoints_.size() == maxNumFusionFrames_)
          publishPointCloud(depthMapPtr, tr, t);
      }
      if (FusionStrategy_ == "CONST_POINTS")
      {
        size_t numFusionPoints = 0;
        for (size_t n = 0; n < dqvDepthPoints_.size(); n++)
          numFusionPoints += dqvDepthPoints_[n].size();
        if (numFusionPoints > 0.5 * maxNumFusionPoints_)
          publishPointCloud(depthMapPtr, tr, t);
      }
    }
  }

  void esvo2_Mapping::publishPointCloud(
      DepthMap::Ptr &depthMapPtr,
      Transformation &tr,
      ros::Time &t)
  {
    sensor_msgs::PointCloud2::Ptr pc_to_publish(new sensor_msgs::PointCloud2);
    Eigen::Matrix<double, 4, 4> T_world_result = tr.getTransformationMatrix();

    pc_->clear();
    pc_->reserve(50000);
    pc_near_->clear();
    pc_near_->reserve(50000);

    double FarthestDistance = 0.0;
    Eigen::Vector3d FarthestPoint;

    for (auto it = depthMapPtr->begin(); it != depthMapPtr->end(); it++)
    {
      Eigen::Vector3d p_world = T_world_result.block<3, 3>(0, 0) * it->p_cam() + T_world_result.block<3, 1>(0, 3);
      pc_->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));

      if (it->p_cam().norm() < visualize_range_)
        pc_near_->push_back(pcl::PointXYZ(p_world(0), p_world(1), p_world(2)));
    }
#ifdef ESVO_CORE_MAPPING_DEBUG
    LOG(INFO) << "The farthest point (p_cam): " << FarthestPoint.transpose();
#endif

    // publish the local 3D map which is used by the tracker.
    if (!pc_->empty())
    {
#ifdef ESVO_CORE_MAPPING_DEBUG
      LOG(INFO) << "<<<<<<<<<(pointcloud)<<<<<<<<" << pc_->size() << " points are published";
#endif
      pcl::toROSMsg(*pc_, *pc_to_publish);
      pc_to_publish->header.stamp = t;
      pc_pub_.publish(pc_to_publish);
    }

    // publish global pointcloud
    if (bVisualizeGlobalPC_)
    {
      if (t.toSec() - t_last_pub_pc_ > visualizeGPC_interval_)
      {
        PointCloud::Ptr pc_filtered(new PointCloud());
        pcl::VoxelGrid<pcl::PointXYZ> sor;
        sor.setInputCloud(pc_near_);
        // sor.setLeafSize(0.03, 0.03, 0.03);// Used in small scale environment.
        sor.setLeafSize(0.3, 0.3, 0.3); // Used in large scale environment.
        sor.filter(*pc_filtered);

        // copy the most current pc tp pc_global
        size_t pc_length = pc_filtered->size();
        size_t numAddedPC = min(pc_length, numAddedPC_threshold_) - 1;
        pc_global_->insert(pc_global_->end(), pc_filtered->end() - numAddedPC, pc_filtered->end());
        // publish point cloud
        pcl::toROSMsg(*pc_global_, *pc_to_publish);
        pc_to_publish->header.stamp = t;
        gpc_pub_.publish(pc_to_publish);
        t_last_pub_pc_ = t.toSec();
      }
    }
  }

  void esvo2_Mapping::publishImage(
      const cv::Mat &image,
      const ros::Time &t,
      image_transport::Publisher &pub,
      std::string encoding)
  {
    if (pub.getNumSubscribers() == 0)
    {
      return;
    }

    std_msgs::Header header;
    header.stamp = t;
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, encoding.c_str(), image).toImageMsg();
    pub.publish(msg);
  }

  void esvo2_Mapping::createEdgeMask(
      std::vector<dvs_msgs::Event *> &vEventsPtr,
      PerspectiveCamera::Ptr &camPtr,
      cv::Mat &edgeMap,
      std::vector<std::pair<size_t, size_t>> &vEdgeletCoordinates,
      bool bUndistortEvents,
      size_t radius)
  {
    size_t col = camPtr->width_;
    size_t row = camPtr->height_;
    int dilate_radius = (int)radius;
    edgeMap = cv::Mat(cv::Size(col, row), CV_8UC1, cv::Scalar(0));
    vEdgeletCoordinates.reserve(col * row);

    auto it_tmp = vEventsPtr.begin();
    while (it_tmp != vEventsPtr.end())
    {
      // undistortion + rectification
      Eigen::Matrix<double, 2, 1> coor;
      if (bUndistortEvents)
        coor = camPtr->getRectifiedUndistortedCoordinate((*it_tmp)->x, (*it_tmp)->y);
      else
        coor = Eigen::Matrix<double, 2, 1>((*it_tmp)->x, (*it_tmp)->y);

      // assign
      int xcoor = std::floor(coor(0));
      int ycoor = std::floor(coor(1));

      for (int dy = -dilate_radius; dy <= dilate_radius; dy++)
        for (int dx = -dilate_radius; dx <= dilate_radius; dx++)
        {
          int x = xcoor + dx;
          int y = ycoor + dy;

          if (x < 0 || x >= col || y < 0 || y >= row)
          {
          }
          else
          {
            edgeMap.at<uchar>(y, x) = 255;
            vEdgeletCoordinates.emplace_back((size_t)x, (size_t)y);
          }
        }
      it_tmp++;
    }
  }

  void esvo2_Mapping::createDenoisingMask(
      std::vector<dvs_msgs::Event *> &vAllEventsPtr,
      cv::Mat &mask,
      size_t row, size_t col)
  {
    cv::Mat eventMap;
    visualizor_.plot_eventMap(vAllEventsPtr, eventMap, row, col);
    cv::medianBlur(eventMap, mask, 3);
  }

  void esvo2_Mapping::extractDenoisedEvents(
      std::vector<dvs_msgs::Event *> &vCloseEventsPtr,
      std::vector<dvs_msgs::Event *> &vEdgeEventsPtr,
      cv::Mat &mask,
      size_t maxNum)
  {
    vEdgeEventsPtr.reserve(vCloseEventsPtr.size());
    for (size_t i = 0; i < vCloseEventsPtr.size(); i++)
    {
      if (vEdgeEventsPtr.size() >= maxNum)
        break;
      size_t x = vCloseEventsPtr[i]->x;
      size_t y = vCloseEventsPtr[i]->y;
      if (mask.at<uchar>(y, x) == 255)
        vEdgeEventsPtr.push_back(vCloseEventsPtr[i]);
    }
  }

  void esvo2_Mapping::getReprojection(std::vector<EventMatchPair> &vEMP, Eigen::Matrix4d T_last_now, std::vector<dvs_msgs::Event *> &vDenoisedEventsPtr_left_dy)
  {
    for (auto event : vDenoisedEventsPtr_left_dy)
    {
      EventMatchPair emp;
      emp.x_left_ << camSysPtr_->cam_left_ptr_->getRectifiedUndistortedCoordinate(event->x, event->y);
      emp.invDepth_ = 0.1;
      emp.lr_depth = 1 / emp.invDepth_;
      emp.x_left_raw_ << event->x, event->y;
      vEMP.push_back(emp);
    }
    Eigen::Matrix3d R_last_now = T_last_now.block(0, 0, 3, 3);
    Eigen::Vector3d t_last_now = T_last_now.block(0, 3, 3, 1);
    Eigen::Matrix3d K = camSysPtr_->cam_left_ptr_->P_.block(0, 0, 3, 3);
    Eigen::Matrix3d K_inv = K.inverse();

    for (int i = 0; i < vEMP.size(); i++)
    {
      Eigen::Vector3d x_last, x_now;
      x_now << vEMP[i].x_left_(0), vEMP[i].x_left_(1), 1;
      double depth = 1 / vEMP[i].invDepth_;
      x_last = (R_last_now * (depth * K_inv * x_now) + t_last_now);
      x_last = K * x_last / x_last(2);
      vEMP[i].x_last_ << x_last(0), x_last(1);
    }
  }

  void esvo2_Mapping::rejectWithF(std::vector<EventMatchPair> &vEMP)
  {
    if (vEMP.size() >= 8)
    {
      ROS_DEBUG("FM ransac begins");
      TicToc t_f;
      vector<cv::Point2f> un_cur_pts(vEMP.size()), un_prev_pts(vEMP.size());
      for (unsigned int i = 0; i < vEMP.size(); i++)
      {
        Eigen::Vector3d tmp_p; // x, y, 1
        tmp_p.x() = vEMP[i].x_left_(0);
        tmp_p.y() = vEMP[i].x_left_(1);
        un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

        tmp_p.x() = vEMP[i].x_right_(0);
        tmp_p.y() = vEMP[i].x_right_(1);
        un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
      }

      vector<uchar> status;
      cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, 1, 0.99, status);
      // Estimation of fundamental matrix using the RANSAC algorithm
      int size_a = vEMP.size();
      int j = 0;
      for (int i = 0; i < int(vEMP.size()); i++)
        if (status[i])
          vEMP[j++] = vEMP[i];
      vEMP.resize(j);
      ROS_INFO("FM ransac: %d -> %lu: %f", size_a, vEMP.size(), 1.0 * vEMP.size() / size_a);
      ROS_INFO("FM ransac costs: %fms", t_f.toc());
    }
  }

} // esvo_core

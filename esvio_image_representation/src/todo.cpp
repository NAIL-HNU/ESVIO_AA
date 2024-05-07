void ImageRepresentation::createImageRepresentationAtTime_hyperthread(const ros::Time& external_sync_time)
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if(!bSensorInitialized_ || !bCamInfoAvailable_)
    return;

  // create exponential-decayed Time Surface map.
  const double decay_sec = decay_ms_ / 1000.0;
  cv::Mat time_surface_map;
  time_surface_map = cv::Mat::zeros(sensor_size_, CV_64F);

  // distribute jobs
  std::vector<Job> jobs(NUM_THREAD_TS);
  size_t num_col_per_thread = sensor_size_.width / NUM_THREAD_TS;
  size_t res_col = sensor_size_.width % NUM_THREAD_TS;
  for(size_t i = 0; i < NUM_THREAD_TS; i++)
  {
    jobs[i].i_thread_ = i;
    jobs[i].pEventQueueMat_ = pEventQueueMat_.get();
    jobs[i].pTimeSurface_ = &time_surface_map;
    jobs[i].start_col_ = num_col_per_thread * i;
    if(i == NUM_THREAD_TS - 1)
      jobs[i].end_col_ = jobs[i].start_col_ + num_col_per_thread - 1 + res_col;
    else
      jobs[i].end_col_ = jobs[i].start_col_ + num_col_per_thread - 1;
    jobs[i].start_row_ = 0;
    jobs[i].end_row_ = sensor_size_.height - 1;
    jobs[i].external_sync_time_ = external_sync_time;
    jobs[i].decay_sec_ = decay_sec;
  }

  // hyper thread processing
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREAD_TS);
  for(size_t i = 0; i < NUM_THREAD_TS; i++)
    threads.emplace_back(std::bind(&ImageRepresentation::thread, this, jobs[i]));
  for(auto& thread:threads)
    if(thread.joinable())
      thread.join();

  // polarity
  if(!ignore_polarity_)
    time_surface_map = 255.0 * (time_surface_map + 1.0) / 2.0;
  else
    time_surface_map = 255.0 * time_surface_map;
  time_surface_map.convertTo(time_surface_map, CV_8U);

  // median blur
  if(median_blur_kernel_size_ > 0)
    cv::medianBlur(time_surface_map, time_surface_map, 2 * median_blur_kernel_size_ + 1);

  // Publish event image
  static cv_bridge::CvImage cv_image;
  cv_image.encoding = "mono8";
  cv_image.image = time_surface_map.clone();

  if(time_surface_mode_ == FORWARD && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_image.header.stamp = external_sync_time;
    time_surface_pub_.publish(cv_image.toImageMsg());
  }

  if (time_surface_mode_ == BACKWARD && bCamInfoAvailable_ && time_surface_pub_.getNumSubscribers() > 0)
  {
    cv_bridge::CvImage cv_image2;
    cv_image2.encoding = cv_image.encoding;
    cv_image2.header.stamp = external_sync_time;
    cv::remap(cv_image.image, cv_image2.image, undistort_map1_, undistort_map2_, CV_INTER_LINEAR);
    time_surface_pub_.publish(cv_image2.toImageMsg());
  }
}

void ImageRepresentation::thread(Job &job)
{
  EventQueueMat & eqMat = *job.pEventQueueMat_;
  cv::Mat& time_surface_map = *job.pTimeSurface_;
  size_t start_col = job.start_col_;
  size_t end_col = job.end_col_;
  size_t start_row = job.start_row_;
  size_t end_row = job.end_row_;
  size_t i_thread = job.i_thread_;

  for(size_t y = start_row; y <= end_row; y++)
    for(size_t x = start_col; x <= end_col; x++)
    {
      dvs_msgs::Event most_recent_event_at_coordXY_before_T;
      if(pEventQueueMat_->getMostRecentEventBeforeT(x, y, job.external_sync_time_, &most_recent_event_at_coordXY_before_T))
      {
        const ros::Time& most_recent_stamp_at_coordXY = most_recent_event_at_coordXY_before_T.ts;
        if(most_recent_stamp_at_coordXY.toSec() > 0)
        {
          const double dt = (job.external_sync_time_ - most_recent_stamp_at_coordXY).toSec();
          double polarity = (most_recent_event_at_coordXY_before_T.polarity) ? 1.0 : -1.0;
          double expVal = std::exp(-dt / job.decay_sec_);
          if(!ignore_polarity_)
            expVal *= polarity;

          // Backward version
          if(time_surface_mode_ == BACKWARD)
            time_surface_map.at<double>(y,x) = expVal;

          // Forward version
          if(time_surface_mode_ == FORWARD && bCamInfoAvailable_)
          {
            Eigen::Matrix<double, 2, 1> uv_rect = precomputed_rectified_points_.block<2, 1>(0, y * sensor_size_.width + x);
            size_t u_i, v_i;
            if(uv_rect(0) >= 0 && uv_rect(1) >= 0)
            {
              u_i = std::floor(uv_rect(0));
              v_i = std::floor(uv_rect(1));

              if(u_i + 1 < sensor_size_.width && v_i + 1 < sensor_size_.height)
              {
                double fu = uv_rect(0) - u_i;
                double fv = uv_rect(1) - v_i;
                double fu1 = 1.0 - fu;
                double fv1 = 1.0 - fv;
                time_surface_map.at<double>(v_i, u_i) += fu1 * fv1 * expVal;
                time_surface_map.at<double>(v_i, u_i + 1) += fu * fv1 * expVal;
                time_surface_map.at<double>(v_i + 1, u_i) += fu1 * fv * expVal;
                time_surface_map.at<double>(v_i + 1, u_i + 1) += fu * fv * expVal;

                if(time_surface_map.at<double>(v_i, u_i) > 1)
                  time_surface_map.at<double>(v_i, u_i) = 1;
                if(time_surface_map.at<double>(v_i, u_i + 1) > 1)
                  time_surface_map.at<double>(v_i, u_i + 1) = 1;
                if(time_surface_map.at<double>(v_i + 1, u_i) > 1)
                  time_surface_map.at<double>(v_i + 1, u_i) = 1;
                if(time_surface_map.at<double>(v_i + 1, u_i + 1) > 1)
                  time_surface_map.at<double>(v_i + 1, u_i + 1) = 1;
              }
            }
          } // forward
        }
      } // a most recent event is available
    }
}

if(representation_mode_ == DiST)
{
TicToc tt;
representation_DiST_.setTo(cv::Scalar(0));
std::map<double, std::pair<int, int> > SortedTimestampCoordinateMap;
std::vector<std::pair<double, std::pair<int, int> > > ts_coord_vec;
ts_coord_vec.reserve(sensor_size_.height * sensor_size_.width);

// Compute DiT (TODO: to hyper threaded)
for(int y=0; y<sensor_size_.height; ++y)
{
for (int x = 0; x < sensor_size_.width; ++x)
{
dvs_msgs::Event ev0;
if (!pEventQueueMat_->getMostRecentEventBeforeT(x, y, external_sync_time, &ev0))
continue;

//        if(ev0.polarity < 0) continue;

int count = 0;
double t_new = 0;
double t_old = DBL_MAX;

for (int dx = -rho_; dx <= rho_; dx++)
for (int dy = -rho_; dy <= rho_; dy++)
{
int ev_rho_x = ev0.x + dx;
int ev_rho_y = ev0.y + dy;
if (ev_rho_x < 0 || ev_rho_x >= sensor_size_.width || ev_rho_y < 0 || ev_rho_y >= sensor_size_.height)
continue;
dvs_msgs::Event ev_rho;
if (pEventQueueMat_->getMostRecentEventBeforeT(ev_rho_x, ev_rho_y, external_sync_time, &ev_rho))
{
//              if(ev_rho.polarity < 0) continue;//TODO

count++;
if(ev_rho.ts.toSec() > t_new)
t_new = ev_rho.ts.toSec();
if(ev_rho.ts.toSec() < t_old)
t_old = ev_rho.ts.toSec();
}
}

//        if(count > 0)
//        {
//          double D = (t_new - t_old) / count;
//          SortedTimestampCoordinateMap.emplace(ev0.ts.toSec() - alpha_ * D, std::make_pair<int, int>(ev0.x, ev0.y));
//          ts_coord_vec.emplace_back(ev0.ts.toSec() - alpha_ * D, std::make_pair<int, int>(ev0.x, ev0.y));
//        }

if(count > 0)
{
double D = (t_new - t_old) / count;
ts_coord_vec.push_back(std::make_pair(ev0.ts.toSec() - alpha_ * D, std::make_pair<int, int>(ev0.x, ev0.y)));
}
}
}

LOG(INFO) << "Compute DiT: " << tt.toc() << " ms.";
tt.tic();

// global sorting and normalization
//    int idx = 1;
//    int max_id = SortedTimestampCoordinateMap.size();
//    auto it = SortedTimestampCoordinateMap.begin();
//    for(; it != SortedTimestampCoordinateMap.end(); it++, idx++)
//    {
//      representation_DiST_.at<double>(it->second.second, it->second.first) = 1.0 * idx / max_id;
////      LOG(INFO) << "DiST( " << it->second.second << ", " << it->second.first << "): " << representation_DiST_.at<double>(it->second.second, it->second.first);
//    }

std::sort(ts_coord_vec.begin(), ts_coord_vec.end());
for(int i = 0; i < ts_coord_vec.size(); i++)
{
representation_DiST_.at<double>(ts_coord_vec[i].second.second, ts_coord_vec[i].second.first)
= 1.0 * i / ts_coord_vec.size();
}
LOG(INFO) << "Compute DiST: " << tt.toc() << " ms.";
}

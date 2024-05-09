# ESVIO_AA: IMU-Aided Event-based Stereo Visual Odometry

## 1. Related Publications

* **[Event-based Stereo Visual Odometry](https://arxiv.org/abs/2007.15548)**, *Yi Zhou, Guillermo Gallego, Shaojie Shen*, arXiv preprint 2020 (under review).
* **[Semi-dense 3D Reconstruction with a Stereo Event Camera](https://arxiv.org/abs/1807.07429)**, *Yi Zhou, Guillermo Gallego, Henri Rebecq, Laurent Kneip, Hongdong Li, Davide Scaramuzza*, ECCV 2018.
* [**IMU-Aided Event-based Stereo Visual Odometry**](http://arxiv.org/abs/2405.04071), Junkai Niu, Sheng Zhong, Yi Zhou, ICRA 2024.

## 2. Abstract

We improve our previous direct pipeline Event-based Stereo Visual Odometry ( refer to the **[ESVO Project Page](https://sites.google.com/view/esvo-project-page/home)** ) in terms of accuracy and efficiency. 

In this work, we achieve a large improvement in trajectory accuracy on the DSEC dataset.

## 3. Installation

We have tested ESVO on machines with the following configurations

* Ubuntu 18.04.5 LTS + ROS melodic + gcc 5.5.0 + cmake (>=3.10) + OpenCV 3.2
* Ubuntu 20.04 LTS + ROS Noetic + OpenCV 4.2

### 3.1 Driver Installation

To work with event cameras, especially for the Dynamic Vision Sensors (DVS/DAVIS), you need to install some drivers. Please follow the instructions (steps 1-9) at [rpg_dvs_ros](https://github.com/uzh-rpg/rpg_dvs_ros) before moving on to the next step. Note that you need to replace the name of the ROS distribution with the one installed on your computer.

We use catkin tools to build the code. You should have it installed during the driver installation.

### 3.2 Dependencies and ESVIO_AA Installation

You should have created a catkin workspace in Section 3.1. If not, please go back and create one.

**Clone this repository** into the `src` folder of your catkin workspace.

```shell
cd ~/catkin_ws/src 
git clone https://github.com/NAIL-HNU/ESVIO_AA.git
```

Then **clone the required dependency** packages

```shell
cd ~/catkin_ws/src
git clone https://github.com/catkin/catkin_simple.git
git clone https://github.com/uzh-rpg/rpg_dvs_ros.git
git clone https://github.com/ethz-asl/gflags_catkin.git
git clone https://github.com/ethz-asl/glog_catkin.git
git clone https://github.com/ethz-asl/minkindr.git 
git clone https://github.com/ethz-asl/eigen_catkin.git
git clone https://github.com/ethz-asl/eigen_checks.git
git clone https://github.com/ethz-asl/minkindr_ros.git
git clone https://github.com/ethz-asl/catkin_boost_python_buildtool.git
git clone https://github.com/ethz-asl/numpy_eigen.git
```

If you don't have a yaml, please install one. But if you already have the yaml library, please do not install it repeatedly, as it will cause version conflicts.

```shell
# if you don't have a yaml
cd ~/catkin_ws/src 
git clone https://github.com/jbeder/yaml-cpp.git
cd yaml-cpp
mkdir build && cd build && cmake -DYAML_BUILD_SHARED_LIBS=ON ..
make -j
```

Finally **compile it**.

```shell
cd ~/catkin_ws
catkin_make
```

## 4. Usage

### 4.1 Time Surface and AA map

Compared with ESVO, we have accelerated the generation speed of time surface. You can try to run it using the following commands.

```shell
cd ~/catkin_ws
source devel/setup.bash
roslaunch esvio_image_representation esvio_image_representation_stereo_AA.launch
```

### 4.2 Running the system on DSEC dataset

Since the rosbag with event and IMU data is not given in the DSEC data set, we package the required data as the input of the system. 

You can get part of the rosbag we repacked through the [zurich_city_04a_download link](https://drive.google.com/file/d/14X_iXLNn25mYKAtqgo3qSTQDtxnmyazs/view?usp=drive_link)  and the [zurich_city_04b_download link](https://drive.google.com/file/d/1-DnOzkxc-aj7whOrdBoxuQUOXhjazoPw/view?usp=drive_link). In addition, if you need repacked packages for other sequences of zurich_city_04 or zurich_city_11, please contact us.

After you get the repackaged data, you can try running it using the following command.

```shell
cd ~/catkin_ws
source devel/setup.bash
roslaunch esvo_core system_dsec.launch
```

## 5. Comparing with us

We welcome comparative evaluation before this project is open sourced. The original trajectories estimated by ESVO ( Block Matching using 10,000 events ) and ESVIO_AA ( Block Matching using 5,000 events ) on DSEC dataset have been uploaded.

Among them, `stamped_groundtruth_alignment.txt` is the trajectory output by the lidar algorithm rotated to the camera system.

## 6. Contact

For questions or inquiries, please feel free to contact us at JunkaiNiu@hnu.edu.cn or eeyzhou@hnu.edu.cn.

We appreciate your interest in our work!


#include <esvio_image_representation/ImageRepresentation.h>

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "esvio_image_representation");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  esvio_image_representation::ImageRepresentation ts(nh, nh_private);
  ros::spin();
  return 0;
}

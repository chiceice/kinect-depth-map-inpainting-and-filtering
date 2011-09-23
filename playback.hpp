// Class to abstract away playing from videos, recorded files or directly from kinect.
#ifndef INCLUDE_NI
#include <XnCppWrapper.h>
#define INCLUDE_NI
#endif

#ifndef INCLUDE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#define INCLUDE_OPENCV
#endif

#include "convert.hpp"

using namespace xn;
using namespace cv;

class KinectPlayback
{
private:
  enum PlayBackType {VIDEO, NODE, DEVICE} mode;
  Context ni_context;
  Player player;
  DepthGenerator g_depth;
  ImageGenerator g_image;
  VideoCapture c_depth;
  VideoCapture c_image;
  int fps;
  int rows;
  int cols;
public:
  // Depth is of type 16UC1. RGB is 24-bit BGR image.
  Mat depth, rgb;
  
  // Constructor.
  KinectPlayback();
  
  // Init from device.
  void init();
  
  // Init from video/depth files.
  void init(const char* rgb_filename, const char* depth_filename);
  
  // Init from kinect node.
  void init(const char* node_filename);
  
  // Get next image and place it into the Mats.
  // Returns true if new image available.
  bool update();
};
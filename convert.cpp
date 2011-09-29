#include "convert.hpp"

using namespace cv;
using namespace xn;


// Convert depth map to a CV_16U image.
void convert_depth_map(const XnDepthPixel* pDepthMap, Mat& cv_depth, int rows, int cols)
{
  cv_depth = Mat(rows, cols, CV_16UC1);
  MatIterator_<unsigned short> it = cv_depth.begin<unsigned short>();
  double sumOriginal = 0;
  double sumFinal = 0;
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      (*it) = pDepthMap[i*cols + j];
      it++;
    }
  }
}

void convert_rgb_map(const XnRGB24Pixel* pImageMap, Mat& cv_depth, int rows, int cols)
{
  cv_depth = Mat(rows, cols, CV_8UC3);
  MatIterator_<Vec3b> it = cv_depth.begin<Vec3b>();
  
  for (unsigned int i = 0; i < rows; i++) {
    for (unsigned int j = 0; j < cols; j++) {
      (*it)[0] = (pImageMap[i*cols + j]).nBlue;
      (*it)[1] = (pImageMap[i*cols + j]).nGreen;
      (*it)[2] = (pImageMap[i*cols + j]).nRed;
      it++;
    }
  }
}
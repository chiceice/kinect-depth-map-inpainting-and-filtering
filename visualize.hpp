#ifndef INCLUDE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#define INCLUDE_OPENCV
#endif

using namespace cv;

void visualize(const Mat& depth, Mat& out_img);
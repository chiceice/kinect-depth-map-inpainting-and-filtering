#include <XnCppWrapper.h>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace xn;

void convert_depth_map(const XnDepthPixel* pDepthMap, Mat& cv_depth, int rows, int cols);

void convert_rgb_map(const XnRGB24Pixel* pImageMap, Mat& cv_depth, int rows, int cols);
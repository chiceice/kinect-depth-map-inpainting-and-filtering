#include <XnCppWrapper.h>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace xn;

void convert_pixel_map(const XnDepthPixel* pDepthMap, Mat& cv_depth, int rows, int cols);

void convert_pixel_map(const XnRGB24Pixel* pImageMap, Mat& cv_depth, int rows, int cols);
#include "visualize.hpp"

using namespace cv;
/* visualize function takes in a 16-bit depth image, and converts it into
 * display-friendly 8-bit format.
 * It does this by assigning to each pixel the intensity value:
 * floor((1 - p_b / p_t) * 255), where p_b is the number of pixels
 * which are closer than the given pixel, and p_t is the total number of
 * pixels.
 */

void visualize(const Mat& depth, Mat& out_img)
{
  depth.convertTo(out_img, CV_8UC1, 0.05f);
}
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
  out_img = Mat(depth.rows, depth.cols, CV_16UC1);

  // Create histogram
  Mat Hist(256*256, 1, CV_64FC1, Scalar::all(0));

  MatConstIterator_<unsigned short> i_it = depth.begin<unsigned short>();

  double total = 0;

  for (; i_it != depth.end<unsigned short>(); i_it++)
    if(*i_it != 0)
    {
      (Hist.at<double>(*i_it))++;
      total ++;
    }

  // Accumulate and normalize histogram.
  double acc = 0;

  for (int i = 0; i < 256*256; i++) {
    acc += Hist.at<double>(i);
    Hist.at<double>(i) = acc;
  }

  Hist = Hist / (total);

  i_it = depth.begin<unsigned short>();
  MatIterator_<unsigned short int> o_it = out_img.begin<unsigned short int>();

  // Perform lookup, and write output to the out image.
  for(; i_it != depth.end<unsigned short>(); i_it++, o_it++)
  {
    if (*i_it == 0)
      *o_it = 0;
    else
      *o_it = 65536 * (1 - Hist.at<double>(*i_it));
  }

  // All done!
}

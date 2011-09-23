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
  out_img = Mat(depth.rows, depth.cols, CV_8UC1);
  
  // Create histogram
  Mat Hist(256*256, 1, CV_32FC1);
  
  MatConstIterator_<unsigned short> i_it = depth.begin<unsigned short>();
  
  float total = 0;
  
  for (; i_it != depth.end<unsigned short>(); i_it++)
  {
    if(*i_it)
    {
      (Hist.at<float>(*i_it))++;
      total++;
    }
  }
  
  // Accumulate and normalize histogram.
  float acc = 0;
  
  for (int i = 0; i < 256*256; i++) {
    acc += Hist.at<float>(i);
    Hist.at<float>(i) = acc;
  }
  
  Hist = Hist / (depth.rows * depth.cols);
  
  i_it = depth.begin<unsigned short>();
  MatIterator_<uchar> o_it = out_img.begin<uchar>();
  
  // Perform lookup, and write output to the out image.
  for(; i_it != depth.end<unsigned short>(); i_it++, o_it++)
  {
    if(*i_it)
      *o_it = 255 * (1 - Hist.at<float>(*i_it));
    else
      *o_it = 0;
  }

  
  // All done!
}
#include "filter.hpp"

using namespace cv;

BilinearFilter::BilinearFilter(int size, double sigma1, double sigma2)
{
  kernel_size = size;
  kernel_xy = getGaussianKernel(2*size+1, sigma1);
  r_sigma = sigma2;
}

// Right now, SSD over the given channels.
double BilinearFilter::find_distance(Vec3b color1, Vec3b color2)
{
  double acc = 0;
  for (int i = 0; i < 3; i++)
    acc += (color1[i]-color2[i])*(color1[i]-color2[i]);
  return acc;
}

Mat BilinearFilter::update(const Mat& rgb, const Mat& depth)
{
  Mat b_rgb;
  Mat b_depth;
    
  Mat out(depth.size(), depth.type());
  
  copyMakeBorder(rgb, b_rgb, kernel_size, kernel_size, kernel_size, kernel_size, BORDER_REFLECT);
  
  copyMakeBorder(depth, b_depth, kernel_size, kernel_size, kernel_size, kernel_size, BORDER_REFLECT);
  
  for (int i = kernel_size; i < b_rgb.rows-kernel_size; i++) {
    for (int j = kernel_size; j < b_rgb.cols-kernel_size; j++) {
      //printf("Calculating root pixel %d, %d.\n", i, j);
      // iterate over the kernel
      double acc_f = 0;
      double acc_w = 0;
      Vec3b pix1 = b_rgb.at<Vec3b>(i,j);
      for (int k = -kernel_size; k < kernel_size+1; k++) {
        for (int l = -kernel_size; l < kernel_size+1; l++) {
          //printf("Calculating kernel offset %d, %d.\n", k, l);
          if(b_depth.at<unsigned short>(i+k, j+l) == 0)
            continue;
          double w_d = kernel_xy.at<double>(kernel_size+k, kernel_size+l);
          Vec3b pix2 = b_rgb.at<Vec3b>(i+k, j+l);
          double w_r = std::exp(-find_distance(pix1, pix2)/(2*r_sigma*r_sigma));
          double w = w_d*w_r;
          acc_w += w;
          acc_f += b_depth.at<unsigned short>(i+k,j+l)*w;
        }
      }
      // Assign value.
      unsigned short result;
      if(acc_w == 0)
        result = 0;
      else
        result = acc_f/acc_w;

      out.at<unsigned short>(i-kernel_size, j-kernel_size) = result;
    }
  }
  
  return out;
}
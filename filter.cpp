#include "filter.hpp"
#include <iostream>
using namespace cv;

BilinearFilter::BilinearFilter(int size, double sigma1, double sigma2)
{
  kernel_size = size;  
  r_sigma = sigma2; //What's this? In relation to other sigma?
  int t = 2;
  t_range = t+1;
    
  create3DBilenearKernel(sigma1);
}

void BilinearFilter::create3DBilenearKernel(double sigma1) {

  //Creates a 3D Matrix MatND, but can be treated as Mat.
  //Dimensions are 2k+1 by 2k+1 by 3. Uses 2 previous images for comparison                                                                                                                               
  //Can increase later. Max size seems to be 16. 
  int dims[] = { 2*kernel_size+1, 2*kernel_size+1, t_range+1};
  threeDMat= MatND(3,dims,CV_32F);
  
  double scale_factor = sigma1/t_range;
  
  //For each time value of T, create a gaussian and copy it into the 3D matrix t while scaling
  //down sigma.
  for(int t = 0; t < t_range; t++) {
    kernel_xy = getGaussianKernel(2*kernel_size+1, sigma1 - t*scale_factor);

    //Copying over elements from Gaussian Kernel to the new matrix at index t
    for(int i = 0; i < 2*kernel_size+1; i++) {
      for(int j = 0; j < 2*kernel_size+1; j++) {
        threeDMat.at<double>(j,i,t) = kernel_xy.at<double>(j,i);
      }
    }
  }

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
          double w_d = threeDMat.at<double>(kernel_size+k, kernel_size+l, 0); //Using new Kernel with now t=0 No prev buffer
          //double w_d = kernel_xy.at<double>(kernel_size+k, kernel_size+l); //Old Kernel
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

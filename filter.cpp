#include "filter.hpp"
#include "stdio.h"
#include <iostream>

using namespace cv;
using boost::math::normal;

BilinearFilter::BilinearFilter(int s_xy, int s_t, double sig_xy, double sig_t, double sig_d, double sig_c)
{
  size_xy = s_xy;
  size_t  = s_t;
  // kernel size = (t+1)x(2*xy+1)x(2*xy+1)
  // so 0,0 would look only at current pixel.

  sigma_xy = sig_xy; // in space
  sigma_t = sig_t;  // in time
  sigma_d = sig_d;  // in depth (label)
  sigma_c = sig_c;  // in color

  createKernel();
}

//---------------------------------KERNEL------------------------------------//

void BilinearFilter::createKernel()
{
  int dims[] = {size_t+1, 2*size_xy+1, 2*size_xy+1};
  kernel= Mat(3,dims,CV_64F);

  // Create the separate gaussians for each dimension.
  Mat gauss_x = getGaussianKernel(2*size_xy+1, sigma_xy, CV_64F);
  Mat gauss_t = getGaussianKernel(2*size_t+1, sigma_t, CV_64F); // We will only access half of this one.

  // Now, fill the matrix with the product of these three gaussians.
  double total = 0;
  for (int t = 0; t < size_t + 1; t++)
    for (int i = 0; i < 2*size_xy + 1; i++)
      for (int j = 0; j < 2*size_xy + 1; j++)
      {
        double value = gauss_t.at<double>(size_t + t) *
                       gauss_x.at<double>(i) *
                       gauss_x.at<double>(j);

        kernel.at<double>(t, i, j) = value;

        total += value;
      }

  // Finally, make sure the kernel sums to one!
  kernel = kernel/total;
}

// Apply the kernel to the (updated) buffer at position i,j
unsigned short BilinearFilter::applyKernel(int i, int j)
{
  // Extract the relevant part of the buffer. i,j point to the corner
  // NOT THE CENTER

  Range ranges[] ={ Range::all(),               // t
                    Range(i, i + 2*size_xy + 1),  // x
                    Range(j, j + 2*size_xy + 1)};  // y

  Mat neighbors_rgb = rgb_buf(ranges).clone();
  Mat neighbors_depth = depth_buf(ranges).clone();

  // Init net loop.
  double net_weight = 0;
  double result = 0;

  // Get center pixel color and depth.
  Vec3b center_color = neighbors_rgb.at<Vec3b>(0, size_xy + 1, size_xy + 1);
  unsigned short center_depth = neighbors_depth.at<unsigned short>(0, size_xy + 1, size_xy + 1);

  // Set up color and depth probability distributions.
  normal dist_color(0, sigma_c);
  normal dist_depth(0, sigma_d);

  // Iterate over the neighborhood
  for (int t = 0; t < size_t+1; t++)
    for (int i = 0; i < 2*size_xy + 1; i++)
      for (int j = 0; j < 2*size_xy + 1; j++) {
        // Retreive the domain kernel.
        double weight_k = kernel.at<double>(t,i,j);
        // Compute the range kernel.
        // Color
        Vec3b pix_color = neighbors_rgb.at<Vec3b>(t, i, j);
        double color_distance = find_distance(center_color, pix_color);
        double weight_c = pdf(dist_color, color_distance);

        // Depth
        unsigned short pix_depth = neighbors_depth.at<unsigned short>(t, i, j);
        double depth_distance = find_distance(center_depth, pix_depth);
        double weight_d = pdf(dist_depth, depth_distance);

        // Accumulate output and net_weight
        net_weight += weight_k * weight_c * weight_d;
        result += pix_depth * weight_k * weight_c * weight_d;
        
        //printf("Weights: %f %f %f\n", weight_k, weight_c, weight_d);
      }
  // Normalize result and convert to depth value.
  unsigned short out = (unsigned short)(result/net_weight);
  //printf("Original depth: %d, Final depth: %d\n", center_depth, out);
  return out;
}

//---------------------------------BUFFERS-----------------------------------//
// Perform a copy of elements from m_from to m_to.
void BilinearFilter::copyTo(const Mat& m_from, Mat& m_to)
{
  if (m_from.total() != m_to.total() || m_from.type() != m_to.type()) {
    printf("Error - CopyTo: matrices do not match.\n");
    exit(1);
  }
  
  if(m_from.type() == CV_8UC3)
  {
    MatConstIterator_<Vec3b> from_it = m_from.begin<Vec3b>();
    MatIterator_<Vec3b> to_it = m_to.begin<Vec3b>();
    for (; from_it != m_from.end<Vec3b>(); from_it++, to_it++)
      *to_it = *from_it;
    return;
  }

  if(m_from.type() == CV_8UC1)
  {
    MatConstIterator_<uchar> from_it = m_from.begin<uchar>();
    MatIterator_<uchar> to_it = m_to.begin<uchar>();
    for (; from_it != m_from.end<uchar>(); from_it++, to_it++)
      *to_it = *from_it;
    return;
  }
  
  if(m_from.type() == CV_16UC1)
  {
    MatConstIterator_<unsigned short> from_it = m_from.begin<unsigned short>();
    MatIterator_<unsigned short> to_it = m_to.begin<unsigned short>();
    for (; from_it != m_from.end<unsigned short>(); from_it++, to_it++)
      *to_it = *from_it;
    return;
  }
  
  if(m_from.type() == CV_32FC1)
  {
    MatConstIterator_<float> from_it = m_from.begin<float>();
    MatIterator_<float> to_it = m_to.begin<float>();
    for (; from_it != m_from.end<float>(); from_it++, to_it++)
      *to_it = *from_it;
    return;
  }
  
  if(m_from.type() == CV_64FC1)
  {
    MatConstIterator_<double> from_it = m_from.begin<double>();
    MatIterator_<double> to_it = m_to.begin<double>();
    for (; from_it != m_from.end<double>(); from_it++, to_it++)
      *to_it = *from_it;
    return;
  }
  
  printf("Error - CopyTo: this matrix type is not supported.\n");
  exit(1);
}

// Init padded buffers for the rgb and depth video.
void BilinearFilter::initBuffers(const Mat& rgb, const Mat& depth)
{
  // Dims should leave room for copymakeborder.
  if(rgb.rows != depth.rows || rgb.cols != depth.cols)
  {
    printf("rgb and depth dimensions must match. \n");
    exit(1);
  }

  int dims[] = {size_t+1, rgb.rows + 2*size_xy, rgb.cols + 2*size_xy};
  rgb_buf = Mat(3, dims, CV_8UC3, Scalar::all(0));
  depth_buf = Mat(3, dims, CV_16UC1, Scalar::all(0));
}

// update the buffers with the current frame.
void BilinearFilter::updateBuffers(const Mat& rgb, const Mat& depth)
{
  // First, shift the buffers back to open up space for new frame.
  Range rangesfrom[] = {Range(0, size_t), Range::all(), Range::all()};
  Mat temp_rgb = rgb_buf(rangesfrom).clone();
  Mat temp_depth = depth_buf(rangesfrom).clone();

  Range rangesto[] = {Range(1, size_t+1), Range::all(), Range::all()};
  Mat rgb_to(rgb_buf(rangesto));
  copyTo(temp_rgb, rgb_to);
  Mat depth_to(depth_buf(rangesto));
  copyTo(temp_depth, depth_to);

  // Now, pad the new frame with copymakeborder.
  Mat rgb_border, depth_border;
  copyMakeBorder(rgb, rgb_border, size_xy, size_xy, size_xy,
                 size_xy, BORDER_REFLECT);

  copyMakeBorder(depth, depth_border, size_xy, size_xy, size_xy, size_xy, BORDER_REFLECT);

  // Finally, copy the new frame into the buffer.
  Range rangesupdate[] = {Range(0,1), Range::all(), Range::all()};
  rgb_to = rgb_buf(rangesupdate);
  copyTo(rgb_border, rgb_to);
  
  depth_to = depth_buf(rangesupdate);
  if(depth_to.type() == depth_buf.type())
    printf("types match\n");
  copyTo(depth_border, depth_to);
  
}

// Right now, SSD over the given channels.
double BilinearFilter::find_distance(Vec3b color1, Vec3b color2)
{
  double acc = 0;
  for (int i = 0; i < 3; i++)
    acc += (color1[i]-color2[i])*(color1[i]-color2[i]);
  return sqrt(acc);
}

double BilinearFilter::find_distance(unsigned short depth1, unsigned short depth2)
{
  return(abs((double)depth1-(double)depth2));
}


// Update the filter and return the filtered result of the current frame.
Mat BilinearFilter::update(const Mat& rgb, const Mat& depth)
{
  if (rgb_buf.empty() || depth_buf.empty())
    initBuffers(rgb, depth);

  printf("Updating buffer...\n");
  updateBuffers(rgb, depth);

  Mat out(depth.size(), depth.type());

  printf("Applying kernel...\n");
  for (int i = 0; i < depth.rows; i++) {
    for (int j = 0; j < depth.cols; j++) {
      out.at<unsigned short>(i,j) = applyKernel(i,j);
    }
  }

  return out;
}

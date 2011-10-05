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

  // Create distros.
  if (sigma_c != 0)
    dist_color = normal(0,sigma_c);
  else
    dist_color = normal(0,1); // dummy value.
  
  if (sigma_d != 0)
    dist_depth = normal(0,sigma_d);
  else
    dist_depth = normal(0,1); // dummy value.

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

  Mat neighbors_rgb = rgb_buf(ranges);
  Mat neighbors_depth = depth_buf(ranges);

  // Init net loop.
  double net_weight = 0;
  double result = 0;

  // Get center pixel color and depth.
  Vec3b center_color = neighbors_rgb.at<Vec3b>(0, size_xy, size_xy);
  unsigned short center_depth = neighbors_depth.at<unsigned short>(0, size_xy , size_xy);

  // Iterate over the neighborhood
  for (int t = 0; t < size_t+1; t++)
    for (int i = 0; i < 2*size_xy + 1; i++)
      for (int j = 0; j < 2*size_xy + 1; j++) {
        // Retreive the domain kernel.
        double weight_k = kernel.at<double>(t,i,j);
        // Compute the range kernel.
        // Color
        double weight_c;
        if (sigma_c != 0) {
          Vec3b pix_color = neighbors_rgb.at<Vec3b>(t, i, j);
          double color_distance = find_distance(center_color, pix_color);
          weight_c = 2*(1-cdf(dist_color, color_distance));
        }

        // Depth
        double weight_d = 1;
        unsigned short pix_depth = neighbors_depth.at<unsigned short>(t, i, j);
        if(sigma_d != 0)
        {
          double depth_distance = find_distance(center_depth, pix_depth);
          weight_d = 2*(1-cdf(dist_depth, depth_distance));
        }

        // Accumulate output and net_weight
        net_weight += weight_k * weight_c * weight_d;
        result += (double)pix_depth * weight_k * weight_c * weight_d;
        
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
  Mat rgb_to, depth_to;
  if (size_t != 0) {
    // First, shift the buffers back to open up space for new frame.
    Range rangesfrom[] = {Range(0, size_t), Range::all(), Range::all()};
    Mat temp_rgb = rgb_buf(rangesfrom).clone();
    Mat temp_depth = depth_buf(rangesfrom).clone();

    Range rangesto[] = {Range(1, size_t+1), Range::all(), Range::all()};
    rgb_to = Mat(rgb_buf(rangesto));
    copyTo(temp_rgb, rgb_to);
    depth_to = Mat(depth_buf(rangesto));
    copyTo(temp_depth, depth_to);
  }
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
    acc += ((double)color1[i]-(double)color2[i])*((double)color1[i]-(double)color2[i]);
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








//------------------------------Median Filter--------------------------------//
// Right now, SSD over the given channels.
double MedianFilter::find_distance(Vec3b color1, Vec3b color2)
{
  double acc = 0;
  for (int i = 0; i < 3; i++)
    acc += ((double)color1[i]-(double)color2[i])*((double)color1[i]-(double)color2[i]);
  return sqrt(acc);
}
  
MedianFilter::MedianFilter(int s_xy, double r_thresh)
{
  size_xy = s_xy;
  rejection_thresh = r_thresh;
}

unsigned short MedianFilter::applyMedian(const Mat& rgb_patch, const Mat& depth_patch)
{
  Vec3b centerColor = rgb_patch.at<Vec3b>(size_xy, size_xy);
  //unsigned short centerDepth = depth_patch.at<unsigned short>(size_xy, size_xy);
  
  int numtokeep = (int)(rgb_patch.total()*rejection_thresh);
  
  // Compute color space distances.
  Mat distances(1,rgb_patch.rows*rgb_patch.cols, CV_64F, Scalar::all(0));
  Mat indexes;
  MatConstIterator_<Vec3b> in_it = rgb_patch.begin<Vec3b>();
  MatIterator_<double> out_it = distances.begin<double>();
  
  for(;in_it != rgb_patch.end<Vec3b>(); in_it++, out_it++)
  {
    *out_it = find_distance(*in_it, centerColor);
  }
  
  // Sort indexes in ascending order.
  sortIdx(distances, indexes, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
  
  Mat depths(1, numtokeep, CV_16UC1, Scalar::all(0));

  MatIterator_<int> i_it = indexes.begin<int>();
  MatIterator_<unsigned short> d_it = depths.begin<unsigned short>();
  
  // Retreive the relevant depths.
  for (; d_it != depths.end<unsigned short>(); i_it++, d_it++)
  {
    int c =  *i_it%depth_patch.cols;
    int r = (*i_it-c)/depth_patch.cols;
    *d_it = depth_patch.at<unsigned short>(r, c);
  }
  // Sort the relevant depths.
  sortIdx(depths, indexes, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

  return depths.at<unsigned short>(indexes.at<int>(numtokeep/2));
}

Mat MedianFilter::update(const Mat& rgb, const Mat& depth)
{
  Mat out = depth.clone();
  Mat rgb_border, depth_border;
  
  copyMakeBorder(rgb, rgb_border, size_xy, size_xy, size_xy,
                 size_xy, BORDER_REFLECT);
  
  copyMakeBorder(depth, depth_border, size_xy, size_xy, size_xy, size_xy, BORDER_REFLECT);
  
  for (int i = 0; i < rgb.rows; i++) {
    for (int j = 0; j < rgb.cols; j++) {
      out.at<unsigned short>(i,j) = applyMedian(rgb_border(Range(i, i+2*size_xy+1), 
                                                           Range(j, j+2*size_xy+1)), 
                                                depth_border(Range(i, i+2*size_xy+1), 
                                                             Range(j, j+2*size_xy+1)));
    }
  }
  
  printf("out size: %d, %d.\n", out.rows, out.cols);
  
  return out;
}
















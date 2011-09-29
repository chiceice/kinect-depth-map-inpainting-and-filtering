#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <math.h>
#include <boost/math/distributions/normal.hpp>

#include <deque>
#include <utility>
#include <vector>

using namespace cv;
using namespace std;
class BilinearFilter
{
private:
  // Parameters
  int size_xy;
  int size_t;
  // kernel size = (t+1)x(2*xy+1)x(2*xy+1)
  // so 0,0 would look only at current pixel.
  
  // For creating the gaussian convolution kernels.
  double sigma_xy; // in space
  double sigma_t;  // in time
  double sigma_d;  // in depth (label)
  double sigma_c;  // in color
  
  // 3d matricees for rgb, depth, and kernel
  Mat depth_buf;
  Mat rgb_buf;
  Mat kernel;
  
  double find_distance(Vec3b color1, Vec3b color2);
  double find_distance(unsigned short depth1, unsigned short depth2);
  
  void createKernel();
  unsigned short applyKernel(int i, int j);
  
  void initBuffers(const Mat& rgb, const Mat& depth);
  void updateBuffers(const Mat& rgb, const Mat& depth);
  
public:
  // Create bilinear filter.
  BilinearFilter(int s_xy, int s_t, double sig_xy, double sig_t, double sig_d, double sig_c);
  
  // Update buffers and return filtered result.
  Mat update(const Mat& rgb, const Mat& depth);
};

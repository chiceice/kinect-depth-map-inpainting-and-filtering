#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <math.h>

#include <deque>
#include <utility>
#include <vector>

using namespace cv;
using namespace std;
class BilinearFilter
{
private:
  // Gaussian convolution kernel.
  Mat kernel_xy;
  int kernel_size;
  Mat threeDMat;
  
  // parameter for how much pixel similarity affects filter.
  double r_sigma;
  double find_distance(Vec3b color1, Vec3b color2);
  
  int t_range;
  
public:
  BilinearFilter(int size, double sigma, double alpha, int t_levels);
  Mat update(const Mat& rgb, const Mat& depth, deque<Mat >& previous_frames);
  void create3DBilenearKernel(double);
};

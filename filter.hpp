#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <math.h>

using namespace cv;
using namespace std;
class BilinearFilter
{
private:
  // Gaussian convolution kernel.
  Mat kernel_xy;
  int kernel_size;
  
  // parameter for how much pixel similarity affects filter.
  double r_sigma;
  double find_distance(Vec3b color1, Vec3b color2);
  
public:
  BilinearFilter(int size, double sigma, double alpha);
  Mat update(const Mat& rgb, const Mat& depth);
};
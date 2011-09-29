#include <XnCppWrapper.h>
#include <opencv2/highgui/highgui.hpp>
#include "convert.hpp"

using namespace cv;
using namespace xn;
int main(int argc, char * argv[])
{
  if(argc < 2)
  {
    printf("Error - no save file provided.\n");
    return 0;
  }
  
  XnStatus nRetVal = XN_STATUS_OK;
  Context context;
  // Initialize context object 
  nRetVal = context.Init();
  // Create Generator nodes
  DepthGenerator depth;
  nRetVal = depth.Create(context);
  
  ImageGenerator color;
  nRetVal = color.Create(context);
  
  // Set it to VGA maps at 30 FPS
  XnMapOutputMode mapMode;
  mapMode.nXRes = XN_VGA_X_RES;
  mapMode.nYRes = XN_VGA_Y_RES;
  mapMode.nFPS = 30;
  nRetVal = depth.SetMapOutputMode(mapMode);
  nRetVal = color.SetMapOutputMode(mapMode);
  
  depth.GetAlternativeViewPointCap().SetViewPoint(color);
  
  // Create recorder node.
  Recorder recorder;
  nRetVal = recorder.Create(context);
  
  nRetVal = recorder.SetDestination(XN_RECORD_MEDIUM_FILE, argv[1]);
  if(nRetVal != XN_STATUS_OK)
  {
    printf("Error - setting file destination to %s failed.\n", argv[1]);
  }
  
  nRetVal = recorder.AddNodeToRecording(depth);
  nRetVal = recorder.AddNodeToRecording(color);
  
  // Make it start generating data 
  nRetVal = context.StartGeneratingAll();
  
  // Main loop 
  while (waitKey(1)!=27) {
    // Wait for new data to be available
    nRetVal = context.WaitAndUpdateAll();
    if (nRetVal != XN_STATUS_OK) {
      printf("Failed updating data: %s\n", xnGetStatusString(nRetVal));
      continue;
    }
    
    // Take current depth map 
    const XnDepthPixel* pDepthMap = depth.GetDepthMap();
    const XnRGB24Pixel* pImageMap = color.GetRGB24ImageMap();
    Mat cv_depth;
    Mat cv_image;
    convert_depth_map(pDepthMap, cv_depth, XN_VGA_Y_RES, XN_VGA_X_RES);
    convert_rgb_map(pImageMap, cv_image, XN_VGA_Y_RES, XN_VGA_X_RES);
    imshow("depth", cv_depth);
    imshow("image", cv_image);
  }
  recorder.Release();
  context.Shutdown();
}
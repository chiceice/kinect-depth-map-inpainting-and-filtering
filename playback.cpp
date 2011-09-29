#include "playback.hpp"
#include <iostream>

using namespace cv;
using namespace xn;

KinectPlayback::KinectPlayback()
{
  // Do nothing here.
}

// Init from device.
void KinectPlayback::init()
{
  mode = DEVICE;
  
  XnStatus nRetVal = XN_STATUS_OK;
  printf("Initializing OpenNI.\n");
  // Initialize context object 
  nRetVal = ni_context.Init();
  
  if (nRetVal != XN_STATUS_OK)
  {
    printf("Failed to initialize OpenNI: %s\n", xnGetStatusString(nRetVal));
    exit(-1);
  }
  
  // Create a DepthGenerator node
  printf("Creating depth and image nodes.\n");
  nRetVal = g_depth.Create(ni_context);
  nRetVal = g_image.Create(ni_context);
  
  // Set it to VGA maps at 30 FPS
  XnMapOutputMode mapMode;
  mapMode.nXRes = XN_VGA_X_RES;
  mapMode.nYRes = XN_VGA_Y_RES;
  mapMode.nFPS = 30;
  
  cols = XN_VGA_X_RES;
  rows = XN_VGA_Y_RES;
  fps = 30;
  
  printf("Modifying output mode.\n");
  nRetVal = g_depth.SetMapOutputMode(mapMode);
  nRetVal = g_image.SetMapOutputMode(mapMode);
  
  g_depth.GetAlternativeViewPointCap().SetViewPoint(g_image);
  
  printf("Start generating...\n");
  // Make it start generating data 
  nRetVal = ni_context.StartGeneratingAll();
  
  rgb = Mat(rows, cols, CV_8UC3);
  depth = Mat(rows, cols, CV_16UC1);
  printf("done setup\n");
}

// Init from video/depth files.
void KinectPlayback::init(const char* rgb_filename, const char* depth_filename)
{
  mode = VIDEO;
  
  rgb = Mat(rows, cols, CV_8UC3);
  depth = Mat(rows, cols, CV_16UC1);
}

// Init from kinect node.
void KinectPlayback::init(const char* node_filename)
{
  printf("Initializing...\n");
  mode = NODE;
  
  XnStatus nRetVal = XN_STATUS_OK;
  
  nRetVal = ni_context.Init();  
  if (nRetVal != XN_STATUS_OK)
  {
    printf("Failed to initialize OpenNI: %s\n", xnGetStatusString(nRetVal));
    exit(-1);
  }
    
  // Open recording 
  printf("Opening node: %s\n", node_filename);
  nRetVal = ni_context.OpenFileRecording(node_filename, player);
  printf("Node opened.\n");
  // TODO: check error code
  // Take the depth node (we assume recording contains a depth node) 
  nRetVal = ni_context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_depth);
  if (nRetVal != XN_STATUS_OK)
  {
    printf("Failed to get depth node: %s\n", xnGetStatusString(nRetVal));
    exit(-1);
  }
  nRetVal = ni_context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_image);
  if (nRetVal != XN_STATUS_OK)
  {
    printf("Failed to get image node: %s\n", xnGetStatusString(nRetVal));
    exit(-1);
  }
    
  XnMapOutputMode mapMode;
  
  g_depth.GetMapOutputMode(mapMode);
  fps = mapMode.nFPS;
  cols = mapMode.nXRes;
  rows = mapMode.nYRes;
  
  rgb = Mat(rows, cols, CV_8UC3);
  depth = Mat(rows, cols, CV_16UC1);
  
  unsigned int nNumFrames = 0;
	nRetVal = player.GetNumFrames(g_depth.GetName(), nNumFrames);
  printf("Init done. File has %d frames.\n", nNumFrames);
}

// Get next image and place it into the Mats.
// Returns true if new image available.
bool KinectPlayback::update()
{
  XnStatus nRetVal = XN_STATUS_OK;
  nRetVal = ni_context.WaitAndUpdateAll();
  if (nRetVal != XN_STATUS_OK) {
    printf("Failed updating playback: %s\n", xnGetStatusString(nRetVal));
    return false;
  }
    
  const XnDepthPixel* pDepthMap = g_depth.GetDepthMap();
  const XnRGB24Pixel* pImageMap = g_image.GetRGB24ImageMap();
  convert_depth_map(pDepthMap, depth, rows, cols);
  convert_rgb_map(pImageMap, rgb, rows, cols);

  return true;
}
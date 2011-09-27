// A demo for playback capabilities.
#include "playback.hpp"
#include "visualize.hpp"
#include "filter.hpp"
#include <opencv2/core/core.hpp>

#include <utility>
#include <deque>

void help()
{
  printf("Demonstrates playback capabilities. To use, type:\n"
         "./demo\n"
         "    -n <node file name> : to use an ni node recording.\n"
         "    -v <rgb file name> <depth file name> : to use 2 video files.\n"
         "    -d : to use the kinect device.\n");
}

int main(int argc, char* argv[])
{
  KinectPlayback playback = KinectPlayback();
  if (argc < 2)
  {
    help();
    return -1;
  }
  
  if(argv[1][0] == '-')
  {
    switch (argv[1][1]) {
      case 'n':
        playback.init(argv[2]);
        break;
      case 'v':
        playback.init(argv[2], argv[3]);
        break;
      case 'd':
        playback.init();
        break;
      default:
        help();
        return -1;
    }
  }
  else {
    help();
    return -1;
  }
  
  int prev_frame_num = 2;
  
  // Playback object should now be open.
  BilinearFilter filter = BilinearFilter(3, 1.5, 1, 2);
  printf("Filter initalized.\n");
  
  //Using Deque to act as queue. Add new frames on front, pop old from back. deque = {newest, ... ,older}
  deque<Mat> previous_frames = deque<Mat>();
  
  while (playback.update() && waitKey(1) != 27) {    
    //Passing Previous frame buffer
    Mat filtered_depth = filter.update(playback.rgb, playback.depth, previous_frames);

    //If Previous frames aren't fully loaded then just keep adding and don't replace.
    if(previous_frames.size() >= prev_frame_num*2) {
      previous_frames.pop_back();
      previous_frames.pop_back();
    }
    
    //Add newest frames to front of stack.
    previous_frames.push_front(playback.rgb);
    previous_frames.push_front(playback.depth);
    
    Mat out_img, filtered_out_img;
    visualize(filtered_depth, filtered_out_img);
    visualize(playback.depth, out_img);
    
    Mat invalid_mask = (out_img == 0);
    Mat inpainted_depth;
    inpaint(out_img, invalid_mask, inpainted_depth, 5, INPAINT_TELEA);
    
    imshow("rgb", playback.rgb);
    imshow("depth", out_img);
    imshow("filtered", filtered_out_img);
    imshow("inpaint", inpainted_depth);
  }
  return 1;
}
# %% importing required libraries
import cv2  # OpenCV library 
import time # time library 
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules 

# %% Opening video feed and defining models

# opening video capture stream
vcap = cv2.VideoCapture(0)
if vcap.isOpened() is False :
  print("[Exiting]: Error accessing webcam stream.")
  exit(0)
fps_input_stream = int(vcap.get(5)) # get fps of the hardware
print("FPS of input stream{}".format(fps_input_stream))
grabbed, frame = vcap.read() # reading single frame for initialization/ hardware warm-up

register_all_modules()
fd_path = '/home/forest/git/pose2sound/mmpose/'

# top-down model setup (from MMPose_Tutorial)
pose_config = fd_path + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
pose_checkpoint = fd_path + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
det_config = fd_path + 'demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

device = 'cuda:0' # check using 'sudo lshw -C display'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

# Quick and dirty alternative 
model = init_model(pose_config, pose_checkpoint, device=device)

# %% Runing real-time inference

# looping as images come in
while True :
    start = time.time()

    grabbed, frame = vcap.read()
    if grabbed is False :
        print('[Exiting] No more frames to read')

    # model inference (aka pose estimation)
    results = inference_topdown(model, frame)

    # showing image overlaid with keypoints   
    
    cv2.imshow('frame' , frame)

    # total loop time
    end = time.time()
    loop_time = (end-start)*1000
    print("Time: {} ms".format(loop_time))

    # user interrupt
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# closing
vcap.release()
cv2.destroyAllWindows()
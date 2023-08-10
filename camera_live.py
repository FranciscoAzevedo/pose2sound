# %%
# importing required libraries
import cv2  # OpenCV library 
import time # time library 

# opening video capture stream
vcap = cv2.VideoCapture(0)
if vcap.isOpened() is False :
  print("[Exiting]: Error accessing webcam stream.")
  exit(0)
fps_input_stream = int(vcap.get(5)) # get fps of the hardware
print("FPS of input stream{}".format(fps_input_stream))
grabbed, frame = vcap.read() # reading single frame for initialization/ hardware warm-up

desired_fps = 60

# processing frames in input stream
num_frames_processed = 0 
start = time.time()
while True :

    grabbed, frame = vcap.read()
    if grabbed is False :
        print('[Exiting] No more frames to read')
        break    # adding a delay for simulating video processing time 

    #delay = 1/desired_fps # delay value in seconds
    time.sleep(delay) 

    num_frames_processed += 1    # displaying frame 
    cv2.imshow('frame' , frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

end = time.time()

# printing time elapsed and fps 
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} ".format(fps, elapsed))# releasing input stream , closing all windows 
vcap.release()
cv2.destroyAllWindows()
# %%

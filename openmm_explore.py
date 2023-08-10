# %% Objective: to process an image as fast as possible
import time
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules  

register_all_modules()

fd_path = '/home/forest/git/pose2sound/mmpose/'

config_file = fd_path + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = fd_path + 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
model = init_model(config_file, checkpoint_file, device='cuda:0')

start = time.time()

# please prepare an image with person
results = inference_topdown(model, fd_path+'demo.jpg')

end = time.time()
speed = (end-start)*1000 # convert to ms
print("Elapsed Time: {} ms".format(speed))
# %%

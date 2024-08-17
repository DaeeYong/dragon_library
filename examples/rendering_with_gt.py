import numpy as np
from dragon import dragonV

video_path = ''
gt_npy_path = ''
openpose_xlsx = ''


all_frame_list =  dragonV.xlsx2data(openpose_xlsx)
gt_npy = np.load(gt_npy_path)
gt_list = gt_npy.tolist()

joint_gt_pair_list = dragonV.make_dataAndGtPair(all_frame_list, gt_list)
dragonV.render_result_on_video(video_path, joint_gt_pair_list, "video")
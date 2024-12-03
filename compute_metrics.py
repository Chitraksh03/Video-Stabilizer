import cv2
import numpy as np
from metrics_tools.interframe_psnr_metric import compute_interframe_psnr
from metrics_tools.global_psnr_metric import compute_global_psnr

def calculate_metrics(original_frames, stabilized_frames, num_frames):
    interframe_psnr_list = []
    global_psnr_list = []

    #inter-frame PSNR for stabilized frames
    interframe_psnr = compute_interframe_psnr(stabilized_frames, num_frames)
    interframe_psnr_list.append(interframe_psnr)

    #global PSNR for each frame in the sequence
    global_psnr = compute_global_psnr(original_frames, stabilized_frames, num_frames)
    global_psnr_list.append(global_psnr)

    #compute averages
    avg_interframe_psnr=float('nan')
    avg_global_psnr=float('nan')
    if(interframe_psnr_list):
        avg_interframe_psnr = np.mean(interframe_psnr_list)
    if(global_psnr_list):
        avg_global_psnr = np.mean(global_psnr_list)

    return avg_interframe_psnr, avg_global_psnr
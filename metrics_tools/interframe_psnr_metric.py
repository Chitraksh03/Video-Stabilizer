import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compute_psnr

#using threadpool to hasten metric calculations
from concurrent.futures import ThreadPoolExecutor


def psnr_calculator(frame_pair):
    #this function calculates the PSNR for a pair of frames
    #the mean squared loss is computed as the mean of square of difference between the two frames
    #this is then used to calulate the PSNR
    frame1, frame2=frame_pair
    epsilon=1e-3        #epsilon prevents division by zero
    frame1=frame1.astype(np.float32)
    frame2=frame2.astype(np.float32) + epsilon 
    return compute_psnr(frame1, frame2, data_range=255)

def compute_interframe_psnr(frames, num_frames):
    #this function calculates and returns the interframe PSNR

    #creating pairs of consecutive frames
    frame_pairs=[]
    for i in range(num_frames - 1):
        current_frame=frames[i]
        next_frame=frames[i + 1]
        
        frame_pair=(current_frame, next_frame)
        frame_pairs.append(frame_pair)
        
    # parallizing using threadpool executor
    with ThreadPoolExecutor() as executor:
        psnr_values=list(executor.map(psnr_calculator, frame_pairs))
    
    average_psnr = np.mean(psnr_values)
    return average_psnr

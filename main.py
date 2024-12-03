
import sys
import os
import cv2
from optimization_tools.l1_optimization import stabilize, writeOutput
from pipeline_tools.frame_generation import generate_frames
from pipeline_tools.feature_extraction import createOptTransformList
from pipeline_tools.video_reconstruction import framesToVideo
from compute_metrics import calculate_metrics

def optimization_ideal(video_name, keyword):
    # print(f"Processing video: {video_name} with keyword: {keyword}")

    #generating frames from video
    frames=generate_frames(video_name)

    N=len(frames)
    #get the F-transforms and frame shape using createOptTransformList function 
    F_transforms,frame_shape=createOptTransformList(frames, N)

    #get B_t transforms from F_t transforms
    B_transforms=stabilize(F_transforms, frame_shape, True, None, cropRatio=0.8)

    #final framed for stabilized video
    new_frames=writeOutput(frames, N, B_transforms, (frame_shape[1], frame_shape[0]), f"./output/{keyword}/",cropRatio=0.8)

    
    #converting these frames to video
    framesToVideo(new_frames, f"./output/{keyword}/", "stabilized_video.mp4")

    #calculating pre-stabilization metrics
    avg_interframe_psnr_pre,avg_global_psnr_pre = calculate_metrics(frames, new_frames, N)

    print("Pre-stabilization metrics:")
    print(f"Avg Interframe PSNR: {avg_interframe_psnr_pre:.2f} dB")
    print(f"Avg Global PSNR: {avg_global_psnr_pre:.2f} dB")
    print()
    #calculating post-stabilization metrics
    avg_interframe_psnr_post,avg_global_psnr_post = calculate_metrics(frames, new_frames, N)

    # Print calculated metrics
    print("Post-stabilization metrics:")
    print(f"Avg Interframe PSNR: {avg_interframe_psnr_post:.2f} dB")
    print(f"Avg Global PSNR: {avg_global_psnr_post:.2f} dB")
    
# Ensure the script is executed from the command line with arguments
if __name__ == "__main__":
    video_path = sys.argv[1]  # Input video path
    keyword = sys.argv[2]  # Keyword for output
    optimization_ideal(video_path, keyword)


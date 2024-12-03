import cv2
import os
import numpy as np


# constructing video from the frames
def framesToVideo(newFrames, folderPath, outputVideoPath, fps=30):
    if not os.path.exists(folderPath): 
        os.makedirs(folderPath) 

    fullPath = folderPath + outputVideoPath

    height, width, layers = newFrames[0].shape
    video = cv2.VideoWriter(fullPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    i = 0 
    for frame in newFrames:
        video.write(frame)

        # frameFilename = folderPath + 'frame_' + str(i) + '.jpg'
        # cv2.imwrite(frameFilename, frame)

        i += 1 

    cv2.destroyAllWindows()
    video.release()


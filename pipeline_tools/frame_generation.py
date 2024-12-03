import cv2
import os

def generate_frames(file_path):

    if not(os.path.exists(file_path)):
        print("Video File does not exist")
        return []

    frames=[]
    cam=cv2.VideoCapture(file_path)

    if not(cam.isOpened()):
        print("Unable to Open File")
        return []
    

    try: 
        if not os.path.exists('data'): 
            os.makedirs('data') 

    except OSError: 
        print ('Error creating directory of data') 
    
    # print("Loading Video...")

    while(True):
        cont, frame=cam.read()

        if(cont):
            frames.append(frame)
        else:
            break

    if(len(frames)==0):
        print("No frames generated")
        return []
      
    # print(len(frames), " frames generated")
    
    return frames



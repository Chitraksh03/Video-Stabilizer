import cv2
import numpy as np

def findKeypointsBetweenFrames(frame1, frame2):
    # Convert frames to grayscale for feature extraction
    grayFrame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayFrame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect features in the first frame
    initialPoints = cv2.goodFeaturesToTrack(grayFrame1, maxCorners=500, qualityLevel=0.01, minDistance=10)

    # Compute optical flow to track features in the second frame
    if initialPoints is not None:
        trackedPoints, status, _ = cv2.calcOpticalFlowPyrLK(grayFrame1, grayFrame2, initialPoints, None)
        initialPoints = initialPoints[status == 1]
        trackedPoints = trackedPoints[status == 1]
    else:
        initialPoints = np.array([[0, 0]])
        trackedPoints = np.array([[0, 0]])

    return initialPoints, trackedPoints

def estimateFrameTransform(frame1, frame2, frameShape):
    # Extract keypoints between frames
    keypoints1, keypoints2 = findKeypointsBetweenFrames(frame1, frame2)
    
    # Estimate the affine transformation
    transformMatrix, _ = cv2.estimateAffine2D(keypoints1, keypoints2)
    return transformMatrix

def estimateOptFrameTransform(frame1, frame2, frameShape):
    keypoints1, keypoints2 = findKeypointsBetweenFrames(frame1, frame2)
    
    # Estimate the transformation with reversed keypoints
    transformMatrix, _ = cv2.estimateAffine2D(keypoints2, keypoints1)
    return transformMatrix

def createTransformList(frames, totalFrames):
    # Initialize an array for transformation parameters
    frameDimensions = frames[0].shape
    transforms = np.zeros((totalFrames - 1, 3), np.float32)

    for i in range(totalFrames - 1):
        transform = estimateFrameTransform(frames[i], frames[i + 1], frameDimensions)
        
        # Extract translation and rotation components
        dx = transform[0, 2]
        dy = transform[1, 2]
        rotationAngle = np.arctan2(transform[1, 0], transform[0, 0])

        # Store the transformations
        transforms[i] = [dx, dy, rotationAngle]
        
    return transforms, frameDimensions

def createOptTransformList(frames, totalFrames):
    frameDimensions = frames[0].shape
    transforms = np.zeros((totalFrames, 3, 3), np.float32)
    transforms[:, :, :] = np.eye(3)

    for i in range(totalFrames - 1):
        transform = estimateOptFrameTransform(frames[i], frames[i + 1], frameDimensions)

        # Update the transformation matrix
        transforms[i + 1, :, :2] = transform.T
        
    return transforms, frameDimensions

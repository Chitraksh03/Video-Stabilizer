import pulp as lpp
import numpy as np
import cv2 as cv

w1 = 4  # weight for the 1st derivative
w2 = 30   # weight for the 2nd derivative
w3 = 150 # weight for the 3rd derivative

N = 6    # for full affine transform (rotation, translation ,scaling ,sheer ,aspect ratio) 

# c = (dx_t, dy_t, a_t, b_t, c_t, d_t)'
c1 = [1, 1, 100, 100, 100, 100]       # for 1st derivative
c2 = c1                               # for 2nd derivative 
c3 = c1                               # for 3rd derivative

# Matrix multiplication of Ft and pt
def transformProduct(Ft, pt, t):
    return [
        pt[t, 0] + Ft[2, 0]*pt[t, 2] + Ft[2, 1]*pt[t, 3],
        pt[t, 1] + Ft[2, 0]*pt[t, 4] + Ft[2, 1]*pt[t, 5],
        Ft[0, 0]*pt[t, 2] + Ft[0, 1]*pt[t, 3],
        Ft[1, 0]*pt[t, 2] + Ft[1, 1]*pt[t, 3],
        Ft[0, 0]*pt[t, 4] + Ft[0, 1]*pt[t, 5],
        Ft[1, 0]*pt[t, 4] + Ft[1, 1]*pt[t, 5]
    ]

# Computes the corner points of the crop window
def getCropWindow(imageShape, cropRatio):
    # for center coordinates of the image
    imgCenterX = round(imageShape[1] / 2)
    imgCenterY = round(imageShape[0] / 2)
    # crop window dimensions
    Width = round(imageShape[1] * cropRatio)
    Height = round(imageShape[0] * cropRatio)
    # upper left corner of the crop window
    cropX = round(imgCenterX - Width / 2)
    cropY = round(imgCenterY - Height / 2)
    # crop window corner points
    return [
        (cropX, cropY),
        (cropX + Width, cropY),
        (cropX, cropY + Height),
        (cropX + Width, cropY + Height)
    ]

def stabilize(Ft, frame_shape, first_window=True, prev_Bt=None, cropRatio=0.9, w1=10, w2=1, w3=100):
    # Create lpp minimization objective
    objective = lpp.LpProblem("stabilize", lpp.LpMinimize)
    # Get the number of frames in sequence to be stabilized
    n_frames = len(Ft)
    # Get corners of crop window
    corner_points = getCropWindow(frame_shape, cropRatio)
    # Slack variables for 1st ,2nd and 3rd derivatives
    e1 = lpp.LpVariable.dicts("e1", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    e2 = lpp.LpVariable.dicts("e2", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    e3 = lpp.LpVariable.dicts("e3", ((i, j) for i in range(n_frames) for j in range(N)), lowBound=0.0)
    # Stabilization parameters for each frame
    p = lpp.LpVariable.dicts("p", ((i, j) for i in range(n_frames) for j in range(N)))
    # Construct objective to be minimized using e1, e2 and e3
    objective += w1 * lpp.lpSum([e1[i, j] * c1[j] for i in range(n_frames) for j in range(N)]) + \
            w2 * lpp.lpSum([e2[i, j] * c2[j] for i in range(n_frames) for j in range(N)]) + \
            w3 * lpp.lpSum([e3[i, j] * c3[j] for i in range(n_frames) for j in range(N)])
    # Apply smoothness constraints on the slack variables e1, e2 and e3 using params p
    for t in range(n_frames - 3):
        # calculating Residual
        MFp_t = transformProduct(Ft[t + 1], p, t + 1)
        MFp_t1 = transformProduct(Ft[t + 2], p, t + 2)
        MFp_t2 = transformProduct(Ft[t + 3], p, t + 3)
        Rt = [MFp_t[j] - p[t, j] for j in range(N)]
        Rt1 = [MFp_t1[j] - p[t + 1, j] for j in range(N)]
        Rt2 = [MFp_t2[j] - p[t + 2, j] for j in range(N)]
        # Apply the smoothness constraints on the slack variables e1, e2 and e3
        for j in range(N):
            objective += -1*e1[t, j] <= Rt[j]
            objective += e1[t, j] >= Rt[j]
            objective += -1 * e2[t, j] <= Rt1[j] - Rt[j]
            objective += e2[t, j] >= Rt1[j] - Rt[j]
            objective += -1 * e3[t, j] <= Rt2[j] - 2*Rt1[j] + Rt[j]
            objective += e3[t, j] >= Rt2[j] - 2*Rt1[j] + Rt[j]
    # Constraints
    for t1 in range(n_frames):
        # Proximity Constraints
        # For a_t
        objective += p[t1, 2] >= 0.9
        objective += p[t1, 2] <= 1.1
        # For b_t
        objective += p[t1, 3] >= -0.1
        objective += p[t1, 3] <= 0.1
        # For c_t
        objective += p[t1, 4] >= -0.1
        objective += p[t1, 4] <= 0.1
        # For d_t
        objective += p[t1, 5] >= 0.9
        objective += p[t1, 5] <= 1.1
        # For b_t + c_t
        objective += p[t1, 3] + p[t1, 4] >= -0.1
        objective += p[t1, 3] + p[t1, 4] <= 0.1
        # For a_t - d_t
        objective += p[t1, 2] - p[t1, 5] >= -0.05
        objective += p[t1, 2] - p[t1, 5] <= 0.05

        # Inclusion Constraints
        for (cx, cy) in corner_points:
            objective += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy >= 0
            objective += p[t1, 0] + p[t1, 2] * cx + p[t1, 3] * cy <= frame_shape[1]
            objective += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy >= 0
            objective += p[t1, 1] + p[t1, 4] * cx + p[t1, 5] * cy <= frame_shape[0]
    # Continuity constraint
    if not first_window:
        objective += p[0, 0] == prev_Bt[2, 0]
        objective += p[0, 1] == prev_Bt[2, 1]
        objective += p[0, 2] == prev_Bt[0, 0]
        objective += p[0, 3] == prev_Bt[1, 0]
        objective += p[0, 4] == prev_Bt[0, 1]
        objective += p[0, 5] == prev_Bt[1, 1]

    objective.solve()
    # update transform Bt
    Bt = np.zeros((n_frames, 3, 3), np.float32)
    Bt[:, :, :] = np.eye(3)

    if objective.status == 1:
        print("Solution converged")
        for i in range(n_frames):
            Bt[i, :, :2] = np.array([[p[i, 2].varValue, p[i, 4].varValue],
                                               [p[i, 3].varValue, p[i, 5].varValue],
                                               [p[i, 0].varValue, p[i, 1].varValue]])
    else:
        print("Error: Linear Programming problem status:", lpp.LpStatus[objective.status])
    return Bt


def writeOutput(frames, numFrames, Bt, frameShape, outputFolderPath, cropRatio):
    
    P = []
    # Scaling factors for crop ratio
    scaleX = 1 / cropRatio
    scaleY = 1 / cropRatio

    scalingMatrix = np.eye(3, dtype=float)
    scalingMatrix[0][0] = scaleX
    scalingMatrix[1][1] = scaleY

    shiftToCenterMatrix = np.eye(3, dtype=float)
    shiftToCenterMatrix[0][2] = -frameShape[0] / 2.0
    shiftToCenterMatrix[1][2] = -frameShape[1] / 2.0

    shiftBackMatrix = np.eye(3, dtype=float)
    shiftBackMatrix[0][2] = frameShape[0] / 2.0
    shiftBackMatrix[1][2] = frameShape[1] / 2.0

    # Process each frame
    for i in range(numFrames):
        # Adjust the transformation for the current frame
        transformMatrix = np.eye(3, dtype=float)
        transformMatrix[:2][:] = Bt[i, :, :2].T
        finalMatrix = shiftBackMatrix @ scalingMatrix @ shiftToCenterMatrix @ np.linalg.inv(transformMatrix)        
        
        # Apply the affine transformation to stabilize the frame
        Pt = cv.warpAffine(frames[i], finalMatrix[:2, :], frameShape)
        P.append(Pt)
    
    return P

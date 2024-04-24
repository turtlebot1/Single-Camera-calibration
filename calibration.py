import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as R

# Specify the checkerboard dimensions
checkerboard_size = (10, 7)  # Change this to your checkerboard's dimensions (corners, not squares)

# Prepare object points (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

# Read images
images = glob.glob('camera_3/*.jpg') #read images from different cameras individually (camera_1, camera_2, camera_3)
camera = 'camera_3'
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Convert rotation vectors to rotation matrices
R_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]

# Convert to quaternions
quaternions = [R.from_matrix(R_mat).as_quat() for R_mat in R_matrices]

# Average quaternions
mean_quaternion = np.mean(quaternions, axis=0)
mean_quaternion /= np.linalg.norm(mean_quaternion)  # Normalize to unit quaternion

# Convert back to rotation matrix
average_R_matrix = R.from_quat(mean_quaternion).as_matrix()

# Average the translation vectors
mean_translation = np.mean(tvecs, axis=0)

# Camera intrinsics
camera_intrinsics = {
    "camera_matrix": mtx,
    "distortion_coefficients": dist
}  

#R_w = np.linalg.inv(average_R_matrix)
M = np.array([[1.0, 0.0, 0.0],
 [0.0, 0.0, -1.0],
  [0.0, 1.0, 0.0]])
  
final_rot = np.array(R_w).dot(M)
#T_w = (-R_w @ mean_translation)  * 10.0

print("camera:", camera)
print("Average Rotation Matrix:", '\n', average_R_matrix)
#print("Average Rotation Matrix:", '\n', final_rot)
print("Average Translation Vector:", mean_translation)
#print("Average Translation Vector:", T_w)
print("Camera Intrinsics:", camera_intrinsics)

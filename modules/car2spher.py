from pprint import pprint

import cv2  # pip install opencv-python
import numpy as np  # pip install numpy
import sys
sys.version
####### CONFIG ########
# CASE = 1
DATA_DIR = 'temp\gaze.csv'
VIDEO_SIZE = [1088, 1080]
NORM = 1
camera_matrix =  np.array([[763.8953424645272,0.0,525.3632286541721],[0.0,763.4849894573791,553.6828720871223],[0.0,0.0,1.0]])
distortion_coeff =  np.array([[0.12620774681640115,0.101277412085089,0.00055471907060676,0.0007493618900665871,0.017274775930009523,0.20475623879702828,0.008992006284546982,0.06396494677510203]])

####### ###### ########


def calc(points_2d):
    # Here we define some example pixel locations. Required shape: Nx2
    
    
    # print("pixel location input:")
    # pprint(points_2d)

    # Secondly, we download the camera intrinsics from Pupil Cloud
 
    # Unproject pixel locations without normalizing. Resulting 3d points lie on a plane
    # with z=1 in reference to the camera origin (0, 0, 0).
    points_3d = unproject_points(
        points_2d, camera_matrix, distortion_coeff, normalize=False
    )
    # print("3d directional output (normalize=False):")
    # pprint(points_3d)

    # Unproject pixel locations with normalizing. Resulting 3d points lie on a sphere
    # with radius=1 around the camera origin (0, 0, 0).
    points_3d = unproject_points(
        points_2d, camera_matrix, distortion_coeff, normalize=True
    )
    # print("3d directional output (normalize=True):")
    # pprint(points_3d)

    radius, elevation, azimuth = cart_to_spherical(points_3d, apply_rad2deg=True)
    # print("radius, elevation, azimuth (in degrees):")
    # elevation: vertical direction
    #   positive numbers point up
    #   negative numbers point bottom
    # azimuth: horizontal direction
    #   positive numbers point right
    #   negative numbers point left
    # convert to numpy array for display purposes:
    res = np.array([elevation, azimuth]).T
    # pprint(res)

    return res



def unproject_points(points_2d, camera_matrix, distortion_coefs, normalize=False):
    """
    Undistorts points according to the camera model.
    :param pts_2d, shape: Nx2
    :return: Array of unprojected 3d points, shape: Nx3
    """
    # print("Unprojecting points...")
    # Convert type to numpy arrays (OpenCV requirements)
    camera_matrix = np.array(camera_matrix)
    distortion_coefs = np.array(distortion_coefs)
    points_2d = np.asarray(points_2d, dtype=np.float32)

    # Add third dimension the way cv2 wants it
    points_2d = points_2d.reshape((-1, 1, 2))

    # Undistort 2d pixel coordinates
    points_2d_undist = cv2.undistortPoints(points_2d, camera_matrix, distortion_coefs)
    # Unproject 2d points into 3d directions; all points. have z=1
    points_3d = cv2.convertPointsToHomogeneous(points_2d_undist)
    points_3d.shape = -1, 3

    if normalize:
        # normalize vector length to 1
        points_3d /= np.linalg.norm(points_3d, axis=1)[:, np.newaxis]

    return points_3d


def cart_to_spherical(points_3d, apply_rad2deg=True):
    # convert cartesian to spherical coordinates
    # source: http://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    # print("Converting cartesian to spherical coordinates...")
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    # elevation: vertical direction
    #   positive numbers point up
    #   negative numbers point bottom
    elevation = np.arccos(y / radius) - np.pi / 2
    # azimuth: horizontal direction
    #   positive numbers point right
    #   negative numbers point left
    azimuth = np.pi / 2 - np.arctan2(z, x)

    if apply_rad2deg:
        elevation = np.rad2deg(elevation)
        azimuth = np.rad2deg(azimuth)

    return radius, elevation, azimuth


if __name__ == "__main__":


    gazes = np.genfromtxt(DATA_DIR, delimiter=',')
    # gazes = tempRead[1:,[3,4]]

    if NORM == 1:
        gazes = gazes * [VIDEO_SIZE[0], VIDEO_SIZE[1]]

    res = calc(gazes)


    np.savetxt("temp/spher.csv", res, delimiter=",")


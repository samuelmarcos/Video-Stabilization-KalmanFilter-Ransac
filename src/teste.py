import sys
from filters import KalmanFilter
from motion_estimator import RansacMotionEstimator
from trajectory import Trajectory
from transformations import fill_mat, get_x, get_y, get_rad_angle, transform
from drawMatches import *
import imutils
import time
import math
import timeit
import os
import glob



import cv2
import numpy as np
from matplotlib import pyplot as plt


ini = time.time()

class FrameInfo:
    def __init__(self):
        self.img = None
        self.img_gray = None
        self.features = []
        self.number = 0
        self.trajectory = Trajectory()


    @property
    def width(self):
        return self.img_gray.shape[1]

    @property
    def height(self):
        return self.img_gray.shape[0]

    @property
    def width_height(self):
        return self.img_gray.shape[::-1]

videoInPath = "../data/MVI_5408.avi"
videoOutPath = "../data/video/"
camera = cv2.VideoCapture(videoInPath)

#videoOutPath = videoInPath +'sab.avi'

N_FRAMES = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = camera.get(cv2.CAP_PROP_FPS)
VID_WIDTH = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
VID_HEIGHT = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print "N_FRAMES: " + str(N_FRAMES)
print "FPS: " + str(FPS)


#camera = cv2.VideoCapture(r"../data/MVI_5408.avi")

#camera = cv2.VideoCapture(r"..\..\data\patio.mp4")
#camera = cv2.VideoCapture(r"..\..\data\1.avi")
# video = cv2.VideoCapture(r"..\..\data\Vietnam_Kim_Long2.avi")
# video = cv2.VideoCapture(r"..\..\data\hippo.mp4")
# video = cv2.VideoCapture(r"..\..\data\videoplayback-cut.mp4")
# video = cv2.VideoCapture(r"C:\Users\Marek\Dropbox\Camera Uploads\Rita\2013-04-24 12.57.54.mp4")
# video = cv2.VideoCapture(r"C:\Users\Marek\Dropbox\Camera Uploads\Rita\2013-04-24 13.12.37.mp4")
# video = cv2.VideoCapture(r"C:\Users\Marek\Dropbox\Camera Uploads\2014-04-27 12.13.45.mp4")


frame = None
prev_frame = None

trajectory = Trajectory(0, 0, 0)
org_trajectories = []
stb_trajectories = []

#frames = []
list_frames = []

crop = 40
crop_rate = crop / 20
filter = KalmanFilter(Trajectory(4e-2, 4e-2, 4e-2), Trajectory(crop_rate, crop_rate, crop_rate), Trajectory(1, 1, 1))
#surf = cv2.SURF(4000)
prev_trans = None
frame_number = 0


frame_width = int(1336.0 / 2)
frame_height = int(768.0 / 2)



def resize(img):
    return cv2.resize(img, (frame_width, frame_height), interpolation=cv2.INTER_LANCZOS4)

lk_params = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.01,
                      minDistance=8,
                      blockSize=19)

crop_rate = 0.9
limits = [int(frame_width * (1 - crop_rate)), int(frame_height * (1 - crop_rate)), 0.05]

#prev_to_cur_transform = []
cont = 0
for k in range(N_FRAMES-1):
    # Capture frame_img-by-frame_img
    ret, frame_img = camera.read()

    if frame_img is None:
        break

    if frame is not None:
        prev_frame = frame

    frame_number += 1

    if frame_number < 220:
        continue

    frame = FrameInfo()
    frame.number = frame_number
    frame.img = frame_img
    frame.img = cv2.resize(frame_img, (0, 0), fx=(1336.0 / frame_img.shape[1]) / 2.0,
                           fy=(768.0 / frame_img.shape[0]) / 2.0, interpolation=cv2.INTER_LANCZOS4)
    frame.img_gray = cv2.cvtColor(frame.img, cv2.COLOR_BGR2GRAY)
    frame.features = cv2.goodFeaturesToTrack(frame.img_gray, **feature_params)
    #frame.features = sift.detect(frame.img_gray,None)
    #frames.append(frame)

    if prev_frame is None:
        continue

    # Optical flow
    new_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_frame.img, frame.img, prev_frame.features, None, **lk_params)
    new_features_for_validation, _, _ = cv2.calcOpticalFlowPyrLK(frame.img, prev_frame.img, new_features, None,
                                                                 **lk_params)

    d = abs(prev_frame.features - new_features_for_validation).reshape(-1, 2).max(-1)
    good = d < 1

    # Select good_features points
    good_new = np.array([x for x, s in zip(new_features, good) if s], dtype=np.float32)
    good_old = np.array([x for x, s in zip(prev_frame.features, good) if s], dtype=np.float32)

    # trans = cv2.estimateRigidTransform(good_old, good_new, fullAffine=False)

    trans, inliers_indices,outliers_indices = RansacMotionEstimator(40, 1.0).estimate(good_old, good_new)

    if trans is None and prev_trans is None:
        print ("wuf? trans is None and prev_trans is none too")
        continue

    if trans is None:
        trans = prev_trans
        print ("wut? trans is None")

    delta = Trajectory(get_x(trans), get_y(trans), get_rad_angle(trans))
    trajectory += delta

    filter.put(trajectory)
    diff = filter.get() - trajectory
    new_delta = delta + diff

    org_trajectories.append(trajectory)
    stb_trajectories.append(filter.get())


    #print >> sys.stderr if abs(get_x(trans)) < 1 and abs(get_y(trans)) < 1 else sys.stdout, (get_x(trans), get_y(trans),), trajectory, new_delta

    # if abs(get_x(trans)) < 1 and prev_trans is not None:
    #     new_delta.x = get_x(prev_trans)
    # if abs(get_y(trans)) < 1 and prev_trans is not None:
    #     new_delta.y = get_y(prev_trans)
    # if abs(get_rad_angle(trans)) < 0.001 and prev_trans is not None:
    #     new_delta.angle = get_rad_angle(prev_trans)

    fill_mat(trans, new_delta.x, new_delta.y, new_delta.angle)

    #prev_to_cur_transform.append([new_delta.x, new_delta.y, new_delta.angle])
    out = cv2.warpAffine(frame.img, trans, frame.width_height, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    #out = cv2.warpAffine(frame_img, trans, frame.width_height, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    #out = frame.img.copy()
    prev_trans = trans.copy()
    cv2.imwrite("../data/video/img%d.png" % cont, out)
    cont += 1


    # # Display the resulting frame_img
    # for t, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = transform(trans, new.ravel())
    #     c, d = transform(trans, old.ravel())
    #     # a, b = new.ravel()
    #     # c, d = old.ravel()
    #
    #     color = [0, 255, 255]
    #     color_bad = [255, 0, 0]
    #     is_inlier = t in inliers_indices
    #     cv2.line(out, (int(a), int(b)), (int(c), int(d)), color if is_inlier else color_bad, 2)
    #     cv2.circle(out, (int(a), int(b)), 3, color if is_inlier else color_bad, -1)
    #
    #
    #
    # def crop_image(img):
    #     return img[crop:frame.height - 2 * crop, crop:frame.width - crop]
    #
    # #out = (crop_image(out))
    # cv2.imshow('out', crop_image(out))
    # cv2.imshow('org', crop_image(frame.img))
    # if cv2.waitKey(1000 / 29) & 0xFF == ord('q'):
    #     break



# plt.plot([f.x for f in org_trajectories], [f.y for f in org_trajectories])
# plt.plot([f.x for f in stb_trajectories], [f.y for f in stb_trajectories])
# plt.show()

# When everything done, release the capture

camera.release()
cv2.destroyAllWindows()

img1 = cv2.imread(videoOutPath)

video = cv2.VideoWriter('video.avi',-1,1,(640,480))
filenames = glob.glob('img/*.png')
for filename in filenames:
    img = cv2.imread(filename)
    video.write(img)

cv2.destroyAllWindows()
video.release()

fim = time.time()
print "Tempo de execucao total, Shi Tomasi: ", str(fim-ini) + " " + "segundos"




############# medicoes ##########################

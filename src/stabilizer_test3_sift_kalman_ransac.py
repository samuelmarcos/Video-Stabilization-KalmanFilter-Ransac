from datetime import datetime
from filters import KalmanFilter
from motion_estimator import RansacMotionEstimator
from stabilzer import OptimalPathStabilizer, OptimalPathStabilizerXYA
from trajectory import Trajectory
from transformations import fill_mat, get_x, get_y, get_rad_angle, transform
from drawMatches import *
#from measurement import get_ITF,read_VideoGray,get_PSNR


import cv2
import numpy as np
import time
import timeit
import math


ini = time.time()
ini4 = time.time()


class FrameInfo:
    def __init__(self):
        # self.img = None
        # self.img_gray = None
        self.features = []
        self.number = 0
        self.trajectory = Trajectory()
        self.shape = ()

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def width_height(self):
        return self.shape[::-1]


video_path = "../data/STREET_original.avi"
video = cv2.VideoCapture(video_path)

N_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = video.get(cv2.CAP_PROP_FPS)
VID_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
VID_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print "N_FRAMES: " + str(N_FRAMES)
print "FPS: " + str(FPS)

frame = None
prev_frame = None

org_trajectories = []
org_transformations = []
frames = []

prev_trans = None
prev_frame_img = None
frame_number = 0

crop = 40
crop_rate = crop / 20
filter = KalmanFilter(Trajectory(4e-2, 4e-2, 4e-2), Trajectory(crop_rate, crop_rate, crop_rate), Trajectory(1, 1, 1))


frame_width = int(1920.0 / 2)
frame_height = int(1080.0 / 2)


def resize(img):
    return cv2.resize(img, (frame_width, frame_height), interpolation=cv2.INTER_LANCZOS4)

lk_params = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# #sift_params = dict(nfeatures = 200,
#                     nOctaveLayers = 3,
#                    contrastThreshold = 0.04,
#                    edgeThreshold = 10,
#                     sigma = 1.6 )


#sift = cv2.xfeatures2d.SIFT_create(**sift_params)
sift = cv2.xfeatures2d.SIFT_create()



motion_estimator = RansacMotionEstimator(20, 1.5, remember_inlier_indices=True)

crop_rate = 0.9
limits = [int(frame_width * (1 - crop_rate)), int(frame_height * (1 - crop_rate)), 0.05]

feature_cont = 0
flow_cont = 0
ransac_cont = 0
kalman_cont = 0
features_quant = []
percent = 0
inliers_quant = []
ouliers_quant = []
compensate_count = 0


#print ("Status: caminho encontrado, iniciando !!")
for k in range(N_FRAMES-1):
    ret, frame_img = video.read()

    if frame_img is None:
        break

    # if frame_number > 25:
    #     break

    if frame is not None:
        prev_frame = frame

    frame_number += 1
    #print ("\rStatus: Processado %00.2f%% (%04d / %04d)") % (frame_number / N_FRAMES * 100, frame_number, N_FRAMES),

    frame = FrameInfo()
    frame.number = frame_number
    frame_img = resize(frame_img)
    frame_img_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)
    #frame.features = orb.detect(frame_img_gray)
    #frame.features = fast.detect(frame_img_gray,None)
    feature_time_ini = time.time()
    frame.features = sift.detect(frame_img_gray,None)
    #frame.features = np.float32([kp.pt for kp in kps])
    #print(type(frame.features))
    #print(frame.features[1])
    #frame.features = cv2.goodFeaturesToTrack(frame_img_gray, **feature_params)
    frame.shape = frame_img_gray.shape
    frames.append(frame)

    features_quant.append(len(frame.features))

#    im2 = cv2.drawKeypoints(frame_img_gray,frame.features)
#    img = cv2.drawKeypoints(frame_img_gray, frame.features, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#    cv2.imshow('point',img)

    if prev_frame is not None:
        # Optical flow
        #calcular o fluxo otico para o movimento
        #passar as duas imagens, os corners para de onde o fluxo sera obtido eum flag
        #status e um vetor dechar que diz se o fluxo de foi achado ou nao para um feature point
        #se for achado tem valor 1

        feature_time_fim = time.time()
        feature_cont += feature_time_fim - feature_time_ini
        feature_time_ini = 0
        feature_time_fim = 0

        flow_time_ini = time.time()
        LK_pts = np.array([[[f.pt[0], f.pt[1]]] for f in frame.features], np.float32)

        new_features, _, _ = cv2.calcOpticalFlowPyrLK(prev_frame_img, frame_img, LK_pts, None, **lk_params)
        new_features_for_validation, _, _ = cv2.calcOpticalFlowPyrLK(frame_img, prev_frame_img, new_features, None,
                                                                     **lk_params)

        flow_time_fim = time.time()
        flow_cont += flow_time_fim - flow_time_ini
        flow_time_ini = 0
        flow_time_fim = 0

        d = abs(LK_pts - new_features_for_validation).reshape(-1, 2).max(-1)
        good_features = d < 1

        # Select good_features points
        good_new = np.array([x for x, s in zip(new_features, good_features) if s], dtype=np.float32)
        good_old = np.array([x for x, s in zip(LK_pts, good_features) if s], dtype=np.float32)


        # trans = cv2.estimateRigidTransform(good_old, good_new, fullAffine=False)
        ransac_time_ini = time.time()
        trans, inliers_indices,outliers_indices = motion_estimator.estimate(good_old, good_new)

        ransac_time_fim = time.time()
        ransac_cont += ransac_time_fim - ransac_time_ini
        ransac_time_ini = 0
        ransac_time_fim = 0

        inliers_quant.append(len(inliers_indices))
        ouliers_quant.append(len(outliers_indices))

        if trans is None and prev_trans is None:
            #print ("wuf? trans is None and prev trans is None")
            continue

        if trans is None:
            #print ("wut? trans is None")
            trans = prev_trans

        org_transformations.append(trans)
        prev_trans = trans.copy()
    prev_frame_img = frame_img

#print ("\nStatus: Caminho encontrado")
#print ("Status: Otimizacao do caminho iniciada")

kalman_time_ini = time.time()
stabilizer = OptimalPathStabilizerXYA(
    [get_x(trans) for trans in org_transformations],
    [get_y(trans) for trans in org_transformations],
    [get_rad_angle(trans) for trans in org_transformations]
    , [limits[0]*0.5, limits[1]*0.5, limits[2]])

new_trans = stabilizer.stabilize()

filter.put(new_trans)
delta = filter.get()

kalman_time_fim = time.time()
kalman_cont += kalman_time_fim - kalman_time_ini
kalman_time_ini = 0
kalman_time_fim = 0

#print ("Status: Caminho otimizado: terminado !!")
#print ("Status: Transformacao do video iniciada")
fim4 = time.time()
ini2 = time.time()

for _ in range(1):
    video.release()
    video = cv2.VideoCapture(video_path)

    videoOutPath = video_path+'sab.avi'

    frame_number = 0
    pressed_q = False
    output_video = cv2.VideoWriter(video_path+'sab.avi',
                                   cv2.VideoWriter_fourcc(*'XVID'), # -1, # int(video.get(cv2.cv.CV_CAP_PROP_FOURCC)),
                                   FPS,
                                   (frame_width - 2*limits[0], frame_height - 2*limits[1])
    )
    trans = np.zeros((2, 3), dtype=np.float32)

    for t, frame in enumerate(frames):
        if t + 1 >= len(org_transformations):
            break

        #print ("\rStatus: Processado %00.2f%% (%04d / %04d)" % (t / N_FRAMES * 100, t, N_FRAMES),

        fill_mat(trans, delta[0][t - 1], delta[1][t - 1], delta[2][t - 1])

        _, frame_img = video.read()
        frame_img = resize(frame_img)

        compensate_time_ini = time.time()
        out = cv2.warpAffine(frame_img, trans, frame.width_height, flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT)

        compensate_time_fim = time.time()
        compensate_count += compensate_time_fim - compensate_time_ini
        compensate_time_ini = 0
        compensate_time_fim = 0

        #mostrar match de poinsts
        for t, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = transform(trans, new.ravel())
            c, d = transform(trans, old.ravel())
            # a, b = new.ravel()
            # c, d = old.ravel()
            color = [0, 255, 255]
            color_bad = [255, 0, 0]
            is_inlier = t in inliers_indices
            cv2.line(out, (int(a), int(b)), (int(c), int(d)), color if is_inlier else color_bad, 2)
            cv2.circle(out, (int(a), int(b)), 3, color if is_inlier else color_bad, -1)


        def crop_image(img):
            return img[limits[1]:frame.height - limits[1], limits[0]:frame.width - limits[0]]

        output_video.write(crop_image(out))
        # cv2.imshow('stab', crop_image(out))
        # cv2.imshow('org', crop_image(frame_img))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            pressed_q = True
            break

    video.release()
    if pressed_q:
        break


    video.release()
    output_video.release()

#print ("Status: Transformacao do video feita")

# for _ in xrange(10):
#     video = cv2.VideoCapture(video_path+'.stab.avi')
#     pressed_q = False
#
#     for t, frame in enumerate(frames):
#         if t + 1 >= len(org_transformations):
#             break
#
#         start_time = datetime.now()
#
#         _, frame_img = video.read()
#
#         if frame_img is None:
#             continue
#
#         cv2.imshow('out', frame_img)
#
#         end_time = datetime.now()
#         elapsed = (end_time - start_time).microseconds / 1e3
#         wait_time = 1000 / video.get(cv2.cv.CV_CAP_PROP_FPS) - elapsed
#
#         if cv2.waitKey(int(wait_time if wait_time > 0 else 1)) & 0xFF == ord('q'):
#             pressed_q = True
#             break
#
#     video.release()
#     if pressed_q:
#         break

print ("Status: feito !!")
cv2.destroyAllWindows()


fim2 = time.time()
fim = time.time()

ini3 = time.time()

soma = 0
soma = sum(features_quant)
percent = soma / len(features_quant)

soma2 = 0
soma2 = sum(inliers_quant)
percent2 = soma /len(inliers_quant)

soma3 = 0
soma3 = sum(outliers_indices)
if len(outliers_indices) != 0:
    percent3 = soma3/len(outliers_indices)
else:
    percent3 = 0

print "-------------------------------------------------------------------------------------------"
print "Quantidade media de pontos detectados por shi-tomasi foi de", str(percent) + " " + "pontos"
print "Quantidade media de inliers foi de ", str(percent2)
print "Quantidade media de outliers foi de ", str(percent3)
print "----------------------------------------------------------------------------------"
print "Tempo para detectar os pontos de interesse foi de", str(feature_cont) + " " + "segundos"
print "Tempo para rastreiar os pontos por fluxo foi de", str(flow_cont) + " " + "segundos"
print "Tempo para executar rancac foi de", str(ransac_cont) + " " + "segundos"
print "Tempo para executar filtro de kalman foi de", str(kalman_cont) + " " + "segundos"
print "-----------------------------------------------------------------------------"
print "Tempo gasto para pre processamento foi de: ", str(fim4-ini4) +  " " + "segundos"
print "Tempo gasto com operacoes secundarias no pre processamento foi de: ", str((fim4-ini4) - (feature_cont + flow_cont + ransac_cont + kalman_cont))  + " " + "segundos"
print "---------------------------------------------------------------------------------"
print "Tempo para estabilizar o video em si: ", str(fim2-ini2) + " " + "segundos"
print "Tempo para compesar os frames foi de: ", str(compensate_count) + " " + "segundos"
print "Tempo gasto em operacoes secundarias na estabilizacao em si foi de: ",  str((fim2-ini2) - compensate_count) + "" + "segundos"
print "----------------------------------------------------------------------------------"
print "Tempo de execucao total, Shi Tomasi e Kalman Filter: ", str(fim-ini) + " " + "segundos"
print "-----------------------------------------------------------------------------------"
############# medicoes ##########################
print "PSNR do video original: " + str(get_ITF(video_path))
print "PSNR do video estabilizado: " + str(get_ITF(videoOutPath))
fim3 = time.time()
print "------------------------------------------------------------------------------------"
print "Tempo gasto para fazer medicoes: " + str(fim3-ini3) + " " + "segundos"

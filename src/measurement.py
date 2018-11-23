import numpy as np
import cv2
from matplotlib import pyplot as plt
from drawMatches import drawMatches
import math


def read_VideoGray (video):
    ret, frame = video.read()
    if ret:
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        frameGray = None
    return ret, frameGray


# Peak Signal to Noise Ratio
def get_PSNR (frame1, frame2):
    mse = np.mean((frame2 - frame1) ** 2 )
    # if mse == 0:
    #     return 100
    PIXEL_MAX = 255.0**2
    return 20*np.log10(PIXEL_MAX / np.sqrt(mse))

# interframe video fidelity
def get_ITF (videoPath):
    video = cv2.VideoCapture(videoPath)
    N_FRAMES = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ITF = 0

    _, currFrame = read_VideoGray(video)
    for i in range(N_FRAMES-1):
        _, nextFrame = read_VideoGray(video)
        ITF += get_PSNR(currFrame, nextFrame)
        currFrame = nextFrame

    ITF = 1.0/(N_FRAMES-1) * ITF
    video.release()
    return ITF

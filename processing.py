import cv2
import numpy as np


def opticalFuckingFlow(current_frame, next_frame, size):
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_RGB2GRAY)
    h, w = size
    # We need three channels cause we'll use HSV
    hsv = np.zeros((h, w, 3))
    # set saturation to 255, we'll drop it before training
    hsv[:, :, 1] = 255
    
    pyr_scale = 0.4
    levels = 1
    winsize = 12
    iterations = 2
    poly_n = 8
    poly_sigma = 1.2

    flow = cv2.calcOpticalFlowFarneback(current_frame, next_frame,
                                        None,
                                        pyr_scale,
                                        levels,
                                        winsize,
                                        iterations,
                                        poly_n,
                                        poly_sigma,
                                        0)

    #optical flow -> Rgb
    #https://gist.github.com/myfavouritekk/2cee1ec99b74e962f816

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[:, :, 0] = angle * (180 / np.pi / 2)
    hsv[:, :, 2] = (magnitude * 20).astype(int)

    hsv = np.asarray(hsv, dtype=np.float32)
    return hsv


def crop_resize_frame(frame, size):
    frame_cropped = frame[25:375, :]  # Remove the useless stuff from the video (Top "vignette" effect + the car itself)

    frame = cv2.resize(frame_cropped, size, interpolation=cv2.INTER_AREA)

    return frame


def process(frame1_path, frame2_path, size, debug=False):
    #Process two adjacent frames
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    frame1_cropped = crop_resize_frame(frame1, size)
    frame2_cropped = crop_resize_frame(frame2, size)

    flow = opticalFuckingFlow(frame1_cropped, frame2_cropped, size)
    if debug:
        return flow, frame1_cropped, frame2_cropped
    return flow

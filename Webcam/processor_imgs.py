import numpy as np
import cv2

def convert_to_input_shape(frame, extra_edge, smaller_edge):
    """Input shape for model is [1, 3, 224, 224] # N C H W.
    Input shape from my webcam is (480, 640, 3) # H W C """
    # convert to square by cropping extra sides. This way, aspect ratio is preserved
    # when resizing to 224, 224.
    # TODO: I added this part to the main code to try to speed it up.
    # frame_shape = (frame.shape[0], frame.shape[1])  # get height and width, ignore the colours channel first
    # extra_edge = (max(frame_shape) - min(frame_shape)) // 2
    if smaller_edge == 'height':  # width < height
        frame = frame[:, extra_edge : -extra_edge]
    elif smaller_edge == 'width':   # width > height
        frame = frame[extra_edge : -extra_edge, :]
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA).astype('float32')
    frame = frame / 255
    frame = frame.transpose((2, 0, 1))  # C H W
    frame = np.expand_dims(frame, 0)    # N C H W

    return frame


def convert_output_to_ndarray(raw_output, d_min=None, d_max=None):
    """Output shape is [1, 1, 224, 224]  # N C H W
    We want a display of (224, 224, 1)  # H W C"""
    raw_output_3D = raw_output.squeeze(0)
    raw_output_HWC = raw_output_3D.transpose((1, 2, 0))
    if d_min is None:
        d_min = np.min(raw_output_HWC)
    if d_max is None:
        d_max = np.max(raw_output_HWC)

    # returns relative depth, can remove this part
    depth_relative = (raw_output_HWC - d_min) / (d_max - d_min)

    return depth_relative

import cv2
import processor_imgs
from run_depth_map_video import init_model, run_model
cap = cv2.VideoCapture(0)
import time


# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# make sure your computer has CUDA enabled, if not, tvm compile the 'scripted_model.pth.tar' for your CPU hardware,
# and the resulting should give u a tar file which u can untar into a directory of your choice and change the below
# directory to that. I compiled for my computer ('scripted_model_p_t-tvm.tar') but it may not fit your hardware.
run, set_input, get_output = init_model('../model/cem_tar_files', cuda=True)   # change this to your model directory
frame_shape = (480, 640)  # get height and width, ignore the colours channel first
extra_edge = (640 - 480) // 2
now = time.time()
count = 0
last_10_frames = []
while True:
    ret, frame = cap.read()
    if frame is None:   # skip iteration if frame not read properly
        print("Frame was skipped")
        continue
    frame = processor_imgs.convert_to_input_shape(frame, frame_shape, extra_edge)
    raw_4d_output = run_model(run, set_input, get_output, frame)    # N C H W
    depth_output = processor_imgs.convert_output_to_ndarray(raw_4d_output)
    last_10_frames.append(depth_output)
    if len(last_10_frames) > 10:
        last_10_frames.pop(0)
    smoothed_output = sum(unit for unit in last_10_frames) / 10

    cv2.imshow('Input', smoothed_output)
    # count += 1
    c = cv2.waitKey(1)
    if c == 27 or count == 100:     # 27 is escape character
        break


# time_taken = time.time() - now
# print("Time taken for 100 frames: ", time_taken)
# print("Time taken for 1 frames: ", time_taken / 100)
# print("Frame rate: ", 1 / (time_taken / 100))
cap.release()
cv2.destroyAllWindows()
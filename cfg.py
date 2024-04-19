number_of_frames = 10
frame_stride = 2
w = 0.75
d = 0.67
r = 1.5
checkpoint = "model_now"
folder_for_checkpoints = "./checkpts"

momentum = 0.09
weight_decay = 0.0005
lr = 0.001

strides_ = [8.0, 16, 32]
box_gain = 7.5
cls_gain = 0.5
dfl_gain = 1.5
reg_max = 16
num_of_cls = 2
batch = 2
iou = 0.7
threshold = 0.5
conf = 0.25
agnostic_nms = False
max_det = 300
amp = True


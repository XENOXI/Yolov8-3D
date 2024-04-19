import yolo_v8_3D
from cfg import number_of_frames,batch,momentum,weight_decay,lr,checkpoint,folder_for_checkpoints,frame_stride,num_of_cls,amp

from yolo_v8_3D import YOLOv8_3D
from yololoss import v8DetectionLoss3D

import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch

table = pd.read_csv("../new_dataset (1).csv")
table.head()

names = table["Video_name"].unique()

def scale(bboxes, w, h):
    """Denormalizes boxes, segments, and keypoints from normalized coordinates."""
    return bboxes[...,:] * torch.FloatTensor((w,h,w,h))

def add_padding(bboxes, padw, padh):   
    return bboxes[...,:] + torch.FloatTensor((padw, padh, padw, padh))


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, img, bboxes):
        """Return updated labels and image with added border."""
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = self.new_shape

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        bboxes = self._update_labels(shape, new_shape, bboxes, ratio, dw, dh)
        return img,bboxes


    def _update_labels(self, shape, new_shape, bboxes, ratio, padw, padh):
        """Update labels."""
        # labels["instances"].convert_bbox(format="xyxy")
        bboxes = scale(bboxes, *shape[::-1])
        bboxes= scale(bboxes, *ratio)
        bboxes = add_padding(bboxes,padw, padh)
        bboxes = scale(bboxes,1/new_shape[0],1/new_shape[0])
        return bboxes
    
class Dataloader():
    def __init__(self) -> None:
        self.vid_id = 0
        self.vid = None
        self.vid_data = None
        self.last_img_data = torch.zeros((number_of_frames,3,640,640),dtype=torch.float32)
        self.frames_cnt = 0
        self.lb = LetterBox(scaleup=False)

        self.i = 0
    def __iter__(self):
        return self
    
    def can_give_data(self):
        global names
        return self.vid_id < names.shape[0]

    def get_one_item(self,batch_id:int):
        global names
        

        data = {"batch_idx":[],"frame":[],"cls":[],"bboxes":[]}
        if self.vid is None:
            self.vid = cv2.VideoCapture(names[self.vid_id])
            #self.vid = cv2.VideoCapture("C:\\Users\\game_\\Downloads\\input1.mp4")
            self.frames_cnt = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.vid_data = table[table["Video_name"] == names[self.vid_id]].sort_values("frame",ascending=True)
            if self.vid_data["frame"].unique().shape[0] < number_of_frames*frame_stride:
                self.vid = None
                return self.get_one_item(batch)
            
            self.last_bbox_data = []
            self.last_cls_data = []

            
            for i in range(number_of_frames):
                for _ in range(frame_stride):
                    _,frame = self.vid.read()
                img,bbox = self.lb(frame,torch.FloatTensor((self.vid_data[self.vid_data["frame"]==i*frame_stride][["min_x", "min_y", "max_x", "max_y"]].to_numpy())))
                self.last_img_data[i] = transforms.ToTensor()(img)
                cls = torch.FloatTensor((self.vid_data[self.vid_data["frame"]==i*frame_stride][["class"]].to_numpy()))

                self.last_bbox_data.append(bbox)
                self.last_cls_data.append(cls)
                for box,cl in zip(bbox,cls): 
                    data["bboxes"].append(box)
                    data["batch_idx"].append(batch_id)
                    data["frame"].append(i)
                    data["cls"].append(cl)
                    
            self.i = number_of_frames * frame_stride
            self.vid_id += 1

            if len(data["bboxes"])!=0:
                data["bboxes"] = torch.stack(data["bboxes"])
                data["batch_idx"] = torch.LongTensor(data["batch_idx"])
                data["cls"] = torch.cat(data["cls"]).long()
                data["frame"] = torch.LongTensor(data["frame"])
            else:
                data["bboxes"] = torch.empty((0,4),dtype = torch.float32)
                data["batch_idx"] = torch.empty((0,),dtype = torch.long)
                data["frame"] = torch.empty((0,),dtype = torch.long)
                data["cls"] = torch.empty((0,),dtype = torch.long)
                
            return self.last_img_data,data

        if self.i+frame_stride >= self.frames_cnt:
            self.vid = None
            return self.get_one_item(batch)
        
        for _ in range(frame_stride):
            _,frame = self.vid.read()

        frames_data = self.vid_data[self.vid_data["frame"]==self.i]
        img,bbox = self.lb(frame,torch.FloatTensor((frames_data[["min_x", "min_y", "max_x", "max_y"]].to_numpy())))

        self.last_img_data[:number_of_frames-1] = self.last_img_data[1:].clone()
        self.last_img_data[-1] = transforms.ToTensor()(img)

        


        self.last_bbox_data.pop(0)
        self.last_cls_data.pop(0)

        self.last_bbox_data.append(bbox)
        self.last_cls_data.append(torch.FloatTensor(frames_data[["class"]].to_numpy()))

        for i in range(number_of_frames):
            for box,cl in zip(self.last_bbox_data[i],self.last_cls_data[i]): 
                data["bboxes"].append(box)
                data["batch_idx"].append(batch_id)
                data["frame"].append(i)
                data["cls"].append(cl)

        if len(data["bboxes"])!=0:
            data["bboxes"] = torch.stack(data["bboxes"])
            data["batch_idx"] = torch.LongTensor(data["batch_idx"])
            data["cls"] = torch.cat(data["cls"]).long()
            data["frame"] = torch.LongTensor(data["frame"])
        else:
            data["bboxes"] = torch.empty((0,4),dtype = torch.float32)
            data["batch_idx"] = torch.empty((0,),dtype = torch.long)
            data["frame"] = torch.empty((0,),dtype = torch.long)
            data["cls"] = torch.empty((0,),dtype = torch.long)

        self.i += frame_stride

        return self.last_img_data,data
    
    def __next__(self):
        if self.vid_id >= names.shape[0]:
            raise StopIteration
        img_data = torch.zeros((batch,number_of_frames,3,640,640),dtype=torch.float32)
        labels = {"batch_idx":[],"frame":[],"cls":[],"bboxes":[]}
        for i in range(batch):
            imgs,data = self.get_one_item(i)
            img_data[i] = imgs
            labels["batch_idx"].append(data["batch_idx"])
            labels["frame"].append(data["frame"])
            labels["cls"].append(data["cls"])
            labels["bboxes"].append(data["bboxes"])
        
        labels["batch_idx"] = torch.cat(labels["batch_idx"])
        labels["frame"] = torch.cat(labels["frame"])
        labels["cls"] = torch.cat(labels["cls"])
        labels["bboxes"] = torch.cat(labels["bboxes"])
        
        if num_of_cls == 3:
            labels["cls"] += 1
            
        return img_data,labels
    
    
import torch.optim as optim

model = YOLOv8_3D()
if checkpoint:
    model.load_state_dict(torch.load(folder_for_checkpoints+"/"+checkpoint))
    
model.train()
model = model.cuda()
loss = v8DetectionLoss3D(model)
optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)


for epoch in range(100):
    j= 0
    p_loss = 0
    dl = Dataloader()

    #x = x.cuda()
    for X,rets in dl:
        while dl.i < 1234:
            x,rets = dl.__next__()
    #while True:
        if j==20:
            for g in optimizer.param_groups:
                g['lr'] = 0.001            
                g['momentum'] = 0.9
            break

        X = X.cuda()
        optimizer.zero_grad()
        
        y = model(X)
        #ls = stl(y[0],torch.ones_like(y[0]))
        #ls.backward()
        #optimizer.step()
        #continue
        ls = loss(y,rets)
        ls[0].backward()

        optimizer.step()
        p_loss += ls[1]
        j+= 1
        if j%20==19:
            p_loss/= 20
            print(f"=< Epoch {epoch} Iter: {j} >= Box loss: {p_loss[0]} || Cls loss: {p_loss[1]} || Dfl loss: {p_loss[2]}")
            torch.save(model.state_dict(), folder_for_checkpoints + f"/model_now")
            p_loss = 0
            
        if j%1000==999:
            torch.save(model.state_dict(), folder_for_checkpoints + f"/model_per_epoch_{epoch}_batch_{j}")
            
            
            
        
    torch.save(model.state_dict(), folder_for_checkpoints + f"/model_per_epoch_{epoch}")
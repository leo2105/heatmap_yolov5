#import numpy as np
#import cv2
#from heatmap_generator import HMap
#import random, time

#cap = cv2.VideoCapture(0)

#hmap = HMap(640, 480, 'stream')
"""
while True:
    ret, img = cap.read()
    if not ret:
        break
    x1, y1 = random.randint(0, 470), random.randint(0, 630)
    x2, y2 = x1, y1
    t = time.time()
    hmap.apply_color_map(x1,y1,x2,y2)
    heatmap_img = hmap.get_heatmap(img)
    print(time.time() -t)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow('hola', heatmap_img)
"""
#############################################################


import argparse
import time
from pathlib import Path
from heatmap_generator import HMap


import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

class Detect_Heat:

    def __init__(self, source):
        ## Detection Initializations
        self.weights, self.view_img, self.save_txt, self.imgsz = 'yolov5s.pt', True, False, 1280
        self.source = source
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))
        self.hmap = HMap(640, 480, 'stream')


        self.view_img = False
        self.save_img = False

        # Initialize
        set_logging()
        self.device = select_device('')
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Set Dataloader
        #vid_path, vid_writer = None, None
        if self.webcam:
            self.view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.imgsz)
        else:
            self.save_img = True
            self.dataset = LoadImages(self.source, img_size=self.imgsz)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        



    def inference(self, img):
        
        pred = self.model(img)[0]

        # Apply NMS
        #                                conf_thres  iou_thres  person=0                                    
        pred = non_max_suppression(pred, 0.25,       0.45,      classes=0)
        return pred


    def detect(self,save_img=False):
        
        for path, img_src, im0s, vid_cap in self.dataset:
            img = img_src.copy()
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.inference(img)
            t2 = time_synchronized()

            input2heatmap = []
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0 = Path(path[i]), '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = Path(path), '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if self.save_img or self.view_img:  # Add bbox to image
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                            aux = [int(x.cpu().detach().numpy()) for x in xyxy]
                            x1, y1, x2, y2 = aux[:]
                            x, y = (x1+x2)//2, y2                            
                            #input2heatmap.append((x,y)) # Pes das pessoas
                            self.hmap.apply_color_map(x-5,y-5,x,y)

                # Print time (inference + NMS)
                #print('%sDone. (%.3fs)' % (s, t2 - t1))

                #self.view_img = False
                # Stream results
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration
        
            #self.add2accumulator(input2heatmap)
            #img_toshow = self.visualization(img_src)

            #hmap.apply_color_map(x1,y1,x2,y2)
            
            img_src = img_src.squeeze()
            img_src = np.moveaxis(img_src, [0,1,2],[2,0,1])
            #print(img_src)
            #print(img_src.shape)
            img_src = cv2.resize(img_src, (640, 480))
            heatmap_img = self.hmap.get_heatmap(img_src)


            cv2.imshow('hola', heatmap_img)
            if cv2.waitKey(1) == ord('q'):
                break

        return img

if __name__ == '__main__':
    with torch.no_grad():
        path = '0'
        detect_heat = Detect_Heat(path)
        detect_heat.detect()

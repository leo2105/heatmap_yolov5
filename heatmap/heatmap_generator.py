import cv2
import numpy as np
import time

class HMap:
    def __init__(self, width, height, name, fade_interval=2):
        self.width = width
        self.height = height
        self.name = name
        self.accum_image = np.zeros((self.height, self.width), np.int32)
        self.accum_image_int = np.zeros((self.height, self.width), np.uint8)
        self.heatmap = np.zeros((self.height, self.width, 3), np.uint8)
        
        self.fade_interval = fade_interval
        self.st = time.time()

        
    def get_mask_from_bbox(self, x, y):
        mask = np.zeros((self.height, self.width), np.uint8)
        radius = 20
        mask = cv2.circle(mask, (x,y), radius, (75,75,75), -1)
        mask = cv2.blur(mask, (105,105), cv2.BORDER_DEFAULT)
        return mask
        
    def apply_color_map(self, x, y):
        # create a mask from image and add it to accum_image
        mask = self.get_mask_from_bbox(x, y).astype(np.int32)
        self.accum_image = cv2.add(self.accum_image, mask)
        aux_img = self.accum_image * 255 // self.accum_image.max()
        self.accum_image_int = np.array(aux_img, dtype=np.uint8)

    def get_heatmap(self, frame):
        self.heatmap = cv2.applyColorMap(self.accum_image_int, cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.7, self.heatmap, 0.5, 0) 
        return frame

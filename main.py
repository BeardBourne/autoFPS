from grabscreen import grab_screen
from models.experimental import attempt_load
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.datasets import letterbox
from utils.plots import plot_one_box
import torch
import numpy as np
import cv2
from numpy import random


with torch.no_grad():
    weights = 'crowdhuman_yolov5m.pt'
    device = select_device('0')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = 640
    model.half()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


    for i in range(100):
        screen = grab_screen(region=(0, 0, 1920, 1080))
        screen = cv2.resize(screen, (imgsz, imgsz))
        im0 = screen
        img = letterbox(im0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)


        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=True)[0]
        pred = non_max_suppression(pred)

        
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    label = f'{names[int(cls)]} {conf:.2f}'

                    if 'head' in label:
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Stream results
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        cv2.imshow('result', im0)











        # cv2.imshow('cv2screen', screen)
        cv2.waitKey(10)
cv2.destroyAllWindows()
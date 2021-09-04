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
import win32api
import win32con
from operator import add, sub

import keys as k
keys = k.Keys()

# from pymouse import PyMouse
import pyautogui

from datetime import datetime

with torch.no_grad():

    # model speed up
    torch.backends.cudnn.benchmark = True

    weights = 'crowdhuman_yolov5m.pt'
    device = select_device('0')
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    windowsz=640
    imgsz = windowsz
    screensz=800
    screen_region = (int((1920-screensz)/2), int((1080-screensz)/2), int((1920+screensz)/2), int((1080+screensz)/2))
    window_region = (int((1920-windowsz)/2), int((1080-windowsz)/2), int((1920+windowsz)/2), int((1080+windowsz)/2))

    center = [int(1920/2), int(1080/2)]

    model.half()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    warmup = True


    while True:
        if win32api.GetAsyncKeyState(win32con.VK_DIVIDE):
            cv2.destroyAllWindows()
            break

        if (win32api.GetAsyncKeyState(win32con.VK_MULTIPLY) or warmup):
            warmup = False
            # print('Loop start')
            # print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

            screen = grab_screen(region=screen_region)
            window = grab_screen(region=window_region)
            im0 = cv2.resize(window, (imgsz, imgsz))
            img = letterbox(im0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)


            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half()
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # print('inference start')
            # print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            pred = model(img, augment=True)[0]
            # print('inference end')
            # print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            pred = non_max_suppression(pred)


            
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                # s = ''
                # s += '%gx%g ' % img.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    head_center = None
                    upper_body_center = None
                    target_center = None

                    for *xyxy, conf, cls in reversed(det):
                        if float(conf) > 0.5:
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                            if 'head' in label:
                                if float(conf) > 0.7:
                                    head_center = [int((c1[0]+ c2[0])/2),int((c1[1]+ c2[1])/2)]
                            else:
                                upper_body_center = [int((c1[0]+ c2[0])/2),int((c1[1]*3+ c2[1])/4)]

                    
                    # reversed so the last xyxy has the most high score.
                    if head_center:
                        target_center = head_center
                        im0 = cv2.circle(im0, head_center, radius=8, color=(255, 0, 0), thickness=-1)
                        head_center = None
                    elif upper_body_center:
                        target_center = upper_body_center
                        im0 = cv2.circle(im0, upper_body_center, radius=10, color=(255, 200, 200), thickness=-1)
                        upper_body_center = None
                    if target_center:



                        # target_center =list(map(add, [window_region[0], window_region[1]], target_center))
                        # delt_to_center = list(map(sub, target_center, center))
                        # print(delt_to_center)
                        # keys.directMouse(delt_to_center[0], delt_to_center[1])
                        # pyautogui.moveRel(delt_to_center[0], delt_to_center[1])
                        


                        win32api.SetCursorPos(list(map(add, [window_region[0], window_region[1]], target_center)))
                        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
                        # win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
                        target_center = None
            # print('SetCursorPos')
            # print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])




            im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            im0 = cv2.resize(im0, (windowsz, windowsz))

            lt = int((screensz-windowsz)/2)
            rb = int((screensz+windowsz)/2)

            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen[lt:rb, lt:rb] = im0

            cv2.rectangle(screen, (lt, lt), (rb, rb), [200, 200, 200], 5, cv2.LINE_AA)
            cv2.imshow('result', screen)
            cv2.waitKey(1)



            
            # print('Loop end')
            # print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
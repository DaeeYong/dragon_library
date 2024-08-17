import cv2
import numpy as np

x0, y0, w, h = -1, -1, -1, -1
thickness = 10
img = None
isDrag = False
windowName = 'img'

def get_roi_pos():
    return x0, y0, w, h

def get_windowName():
     return windowName

def set_WindowName(name:str):
     global windowName
     windowName = name

def set_Img(input_img:np.ndarray):
    global img
    img = input_img

def set_thickness(num):
    global thickness
    thickness = num

def on_Mouse_roi_without_window(event, x, y, flags, param):
    global x0, y0, w, h, isDrag
    if event == cv2.EVENT_LBUTTONDOWN:
        isDrag = True
        x0 = x
        y0 = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if isDrag:
            img_tmp = img.copy()
            cv2.rectangle(img_tmp, (x0, y0), (x, y), (0, 255, 0), thickness)
            cv2.imshow(windowName, img_tmp)

    elif event == cv2.EVENT_LBUTTONUP:
        isDrag = False
        w = x - x0
        h = y - y0
        print(f'x0:{x0}, y0:{y0}, w:{w}, h:{h}')
        if w > 0 and h > 0:
            img_tmp = img.copy()
            cv2.rectangle(img_tmp, (x0, y0), (x, y), (0, 0, 255), thickness)
            cv2.imshow(windowName, img_tmp)

def on_Mouse_roi(event, x, y, flags, param):
        global x0, y0, w, h, isDrag
        if event == cv2.EVENT_LBUTTONDOWN:
            isDrag = True
            x0 = x
            y0 = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if isDrag:
                img_tmp = img.copy()
                cv2.rectangle(img_tmp, (x0, y0), (x, y), (0, 255, 0), thickness)
                cv2.imshow(windowName, img_tmp)


        elif event == cv2.EVENT_LBUTTONUP:
            isDrag = False
            w = x - x0
            h = y - y0
            print(f'x0 :{x0}, y0:{y0}, w:{w}, h:{h}')
            if w > 0 and h > 0:
                img_tmp = img.copy()

                cv2.rectangle(img_tmp, (x0, y0), (x, y), (0, 0, 255), thickness)
                cv2.imshow(windowName, img_tmp)
                roi = img[y0: y0 + h, x0:x0 + w]
                cv2.imshow('roi', roi)
                cv2.moveWindow('roi', 0, 0)
            else:
                cv2.imshow(windowName, img)
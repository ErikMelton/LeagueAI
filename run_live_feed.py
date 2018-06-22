import time
import cv2
import mss
import numpy as np
import argparse
import colorsys
import imghdr
import os
import threading, queue
import random
import keyboard
import win32api, win32con, win32gui, win32ui

from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont
from subprocess import call
from YAD2K.yad2k.models.keras_yolo import yolo_eval, yolo_head
from retrain_yolo import create_model

loaded = False

parser = argparse.ArgumentParser(
    description='Run a YOLO_v2 style detection model. Choose')

parser.add_argument(
    '-s',
    '--score_threshold',
    type=float,
    help='threshold for bounding box scores, default .3',
    default=.3)

parser.add_argument(
    '-iou',
    '--iou_threshold',
    type=float,
    help='threshold for non max suppression IOU, default .5',
    default=.5)

args = parser.parse_args()

def test_yolo(image,is_fixed_size,model_image_size,sess,boxes,scores,classes,yolo_model,input_image_shape,class_names,colors):
    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} boxes'.format(len(out_boxes)))

    # Write data to a JSON file located within the 'output/' directory.
    data = []

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        coords = (left,top,right,bottom)
        data.append(coords)
    
def run_nn(hWnd):
    global loaded

    model_path = os.path.expanduser('YAD2K/model_data/yolo.h5')
    anchors_path = os.path.expanduser('YAD2K/model_data/yolo_anchors.txt')
    classes_path = os.path.expanduser('YAD2K/model_data/league_classes.txt')

    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    yolo_model, _ = create_model(anchors, class_names)
    yolo_model.load_weights('trained_stage_3_best.h5')

    # Verify model, anchors, and classes are compatible
    num_classes = len(class_names)
    num_anchors = len(anchors)
    # TODO: Assumes dim ordering is channel last
    model_output_channels = yolo_model.layers[-1].output_shape[-1]
    assert model_output_channels == num_anchors * (num_classes + 5), 'Mismatch between model and given anchor and class sizes. ' 
    print('{} model, anchors, and classes loaded.'.format(model_path))

    # Check if model is fully convolutional, assuming channel last order.
    model_image_size = yolo_model.layers[0].input_shape[1:3]
    is_fixed_size = model_image_size != (None, None)

    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # Generate output tensor targets for filtered bounding boxes.
    # TODO: Wrap these backend operations with Keras layers.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(yolo_outputs,input_image_shape,score_threshold=args.score_threshold,iou_threshold=args.iou_threshold)

    # Save the output into a compact JSON file.
    outfile = open('output/game_data.json', 'w')
    # This will be appended with an object for every frame.
    data_to_write = []

    loaded = True
    win32gui.RedrawWindow(hWnd, None, None, win32con.RDW_INVALIDATE | win32con.RDW_ERASE)

    with mss.mss() as sct:
        monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
        while 'Screen capturing':
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor))
            img = Image.fromarray(img)
            img.load()
            background = Image.new("RGB",img.size,(255,255,255))
            background.paste(img,mask=img.split()[3])

            test_yolo(background,is_fixed_size,model_image_size,sess,boxes,scores,classes,yolo_model,input_image_shape,class_names,colors)

            # img = test_yolo(background)
            # basewidth = 700
            # wpercent = (basewidth/float(img.size[0]))
            # hsize = int((float(img.size[1])*float(wpercent)))
            # img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            # img = np.array(img)
            # # Display the picture
            # cv2.imshow('OpenCV/Numpy normal', img)
            print('fps: {0}'.format(1 / (time.time()-last_time)))
            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord(';'):
                cv2.destroyAllWindows()
                break

    sess.close()
    return

def wndProc(hWnd, message, wParam, lParam):
    print('Updating')
    if message == win32con.WM_PAINT:
        hdc, paintStruct = win32gui.BeginPaint(hWnd)

        dpiScale = win32ui.GetDeviceCaps(hdc, win32con.LOGPIXELSX) / 60.0
        fontSize = 80

        # http://msdn.microsoft.com/en-us/library/windows/desktop/dd145037(v=vs.85).aspx
        lf = win32gui.LOGFONT()
        lf.lfFaceName = "Times New Roman"
        lf.lfHeight = int(round(dpiScale * fontSize))
        #lf.lfWeight = 150
        # Use nonantialiased to remove the white edges around the text.
        # lf.lfQuality = win32con.NONANTIALIASED_QUALITY
        hf = win32gui.CreateFontIndirect(lf)
        win32gui.SelectObject(hdc, hf)

        rect = win32gui.GetClientRect(hWnd)
        # http://msdn.microsoft.com/en-us/library/windows/desktop/dd162498(v=vs.85).aspx
        if not loaded:
            win32gui.DrawText(hdc,'Loading Neural Net...',-1,rect,win32con.DT_CENTER | win32con.DT_NOCLIP | win32con.DT_SINGLELINE | win32con.DT_VCENTER)
        
        color = win32api.RGB(0,0,255)
        brush = win32gui.CreateSolidBrush(color)
        # for rect in data:
        #     win32gui.FrameRect(hdc,rect,brush)

        win32gui.EndPaint(hWnd, paintStruct)
        return 0

    elif message == win32con.WM_DESTROY:
        print('Closing the window.')
        win32gui.PostQuitMessage(0)
        return 0

    else:
        return win32gui.DefWindowProc(hWnd, message, wParam, lParam)

def setup_screen():
    hInstance = win32api.GetModuleHandle()
    className = 'MyWindowClassName'

    wndClass = win32gui.WNDCLASS()
    wndClass.style = win32con.CS_HREDRAW | win32con.CS_VREDRAW
    wndClass.lpfnWndProc = wndProc
    wndClass.hInstance = hInstance
    wndClass.hCursor = win32gui.LoadCursor(None, win32con.IDC_ARROW)
    wndClass.hbrBackground = win32gui.GetStockObject(win32con.WHITE_BRUSH)
    wndClass.lpszClassName = className

    wndClassAtom = win32gui.RegisterClass(wndClass)
    # http://msdn.microsoft.com/en-us/library/windows/desktop/ff700543(v=vs.85).aspx
    # Consider using: WS_EX_COMPOSITED, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT
    # The WS_EX_TRANSPARENT flag makes events (like mouse clicks) fall through the window.
    exStyle = win32con.WS_EX_COMPOSITED | win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOPMOST | win32con.WS_EX_TRANSPARENT

    # http://msdn.microsoft.com/en-us/library/windows/desktop/ms632600(v=vs.85).aspx
    # Consider using: WS_DISABLED, WS_POPUP, WS_VISIBLE
    style = win32con.WS_DISABLED | win32con.WS_POPUP | win32con.WS_VISIBLE

    # http://msdn.microsoft.com/en-us/library/windows/desktop/ms632680(v=vs.85).aspx
    hWindow = win32gui.CreateWindowEx(exStyle,wndClassAtom,None, # WindowName
        style,
        0, # x
        0, # y
        win32api.GetSystemMetrics(win32con.SM_CXSCREEN), # width
        win32api.GetSystemMetrics(win32con.SM_CYSCREEN), # height
        None, # hWndParent
        None, # hMenu
        hInstance,
        None # lpParam
    )

    # http://msdn.microsoft.com/en-us/library/windows/desktop/ms633540(v=vs.85).aspx
    win32gui.SetLayeredWindowAttributes(hWindow, 0x00ffffff, 255, win32con.LWA_COLORKEY | win32con.LWA_ALPHA)

    # http://msdn.microsoft.com/en-us/library/windows/desktop/dd145167(v=vs.85).aspx
    #win32gui.UpdateWindow(hWindow)

    # http://msdn.microsoft.com/en-us/library/windows/desktop/ms633545(v=vs.85).aspx
    win32gui.SetWindowPos(hWindow, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOACTIVATE | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)

    # http://msdn.microsoft.com/en-us/library/windows/desktop/ms633548(v=vs.85).aspx
    #win32gui.ShowWindow(hWindow, win32con.SW_SHOW)
    t1 = threading.Thread(target=run_nn, args=(hWindow,))
    t1.setDaemon(False)
    t1.start()

    win32gui.PumpMessages()

def _main():
    data = []
    g_hWnd = -1

    setup_screen()
    # t1 = threading.Thread(target=run_nn, args=(args.score_threshold,args.iou_threshold,g_hWnd,data,loaded))
    # t2 = threading.Thread(target=setup_screen, args=(g_hWnd,data,loaded))
    # t1.start()
    # t2.start()

    # t1.join()
    # t2.join()

if __name__ == '__main__':
    _main()
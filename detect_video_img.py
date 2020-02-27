import os
import cv2
import time
import json
import numpy as np

def get_video():
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(1152, 720))
    img = []
    box_all = []
    with open('./build/tests/frame_info.txt', 'r') as f:
        for line in f:
            tmp_box = []
            img_box = line.strip().split(' ')
            if len(img_box) < 2:
                img_pth = img_box[0]
                tmp_box = [[0,0,0,0]]
                img.append(img_pth)
            else:
                img_pth = img_box[0]
                img.append(img_pth)
                boxs = img_box[1].split('|')
                for box in boxs:
                    if box == '':
                        continue
                    tmp_box.append([float(val) for val in box.split(',')])
            box_all.append(tmp_box)
    print('len img = ', len(img))
    print('len box_all = ', len(box_all))
    for box in box_all:
        print('len box = ', len(box))
    for i in range(len(img)):
        frame = cv2.imread(img[i])
        for box in box_all[i]:
            cv2.rectangle(frame,
                    (int(box[0]*1152),int(box[1]*720)),(int(box[2]*1152),int(box[3]*720)),
                    (0, 255, 0), 2)
        videoWriter.write(frame)
    videoWriter.release()

def get_img_mks_box():
    boxes = [[0.288836*1080, 0.628388*1920, 0.508728*1080, 0.751794*1920],
            [0.708191*1080, 0.564356*1920, 0.855852*1080, 0.647229*1920]]
    mks1 = [[int(0.342383*1080),int(0.661245*1920)],
            [int(0.435497*1080),int(0.668438*1920)],
            [int(0.385626*1080),int(0.698722*1920)],
            [int(0.338389*1080),int(0.7237*1920)],
            [int(0.407276*1080),int(0.728452*1920)]]
    mks2 = [[int(0.743102*1080),int(0.590994*1920)],
            [int(0.798637*1080),int(0.591913*1920)],
            [int(0.757151*1080),int(0.611872*1920)],
            [int(0.751201*1080),int(0.628628*1920)],
            [int(0.793066*1080),int(0.629701*1920)]]
    mks = []
    mks.append(mks1)
    mks.append(mks2)
    #box = [0.076156*1080, 0.606703*1920, 0.270785*1080, 0.730839*1920]
    img_pth = "/world/data-gpu-94/fenghui/small_model/facebox_128_228_lite/228-128.jpg"
    frame = cv2.imread(img_pth)
    for i, box in enumerate(boxes):
        cv2.rectangle(frame, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), (0, 255, 0), 2)
        for mk in mks[i]:
            cv2.circle(frame, tuple(mk), 2, (255, 0, 0), -1)
    cv2.imwrite('1.jpg', frame)

def get_img_box():
    badcase_box = '/world/data-gpu-94/fenghui/mhg/test_lst_box.lst'
    vis_dir = '/world/data-gpu-94/fenghui/mhg/mks_vis_curve/inference/test'
    with open(badcase_box, 'r') as f:
        img_info_dict = json.load(f)
    for img_path, values in img_info_dict.items():
        frame = cv2.imread(img_path)
        for box in values['facebox']:
                cv2.rectangle(frame, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(vis_dir, str(time.time()) + '.jpg'), frame)


#get_video()
get_img_box()

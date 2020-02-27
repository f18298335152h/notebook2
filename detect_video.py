import numpy as np
import cv2

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
get_video()

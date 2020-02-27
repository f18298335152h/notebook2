import numpy as np
import cv2
from copy import deepcopy
img_path = []
per_id = []
box = []
fps = 1
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(1920, 1080))
with open('./frame_info.txt', 'r') as f:
    for line in f:
        arr = line.strip().split(' ')
        img_path.append(arr[0])
        box_id = arr[1].split(',')
        box.append([float(box_id[0]), float(box_id[1]), float(box_id[2]),
            float(box_id[3])])
        per_id.append(box_id[4])


boxess = []
#img_idd = []
per_idd = []
img_paths = []
repeat = []
img_p = deepcopy(img_path)
img_pp = []

for i in range(len(img_p)):
    if img_p[i] not in img_pp:
        img_pp.append(img_p[i])

for i in range(len(img_pp)):
    repeat.append([x for x in range(len(img_path)) if img_path[x] == img_pp[i]])

for ids in repeat:
    img_paths.append(img_path[ids[0]])
    tmp_b = []
    tmp_per = []
    for ind in ids:
#        print('ind = ', ind)
#        print(' box = ', boxs[ind])
#        print(' per id = ', per_id[ind])
        tmp_b.append(box[ind])
        tmp_per.append(per_id[ind])

    if len(tmp_b) !=0 and len(tmp_per) !=0:
        boxess.append(tmp_b)
        per_idd.append(tmp_per)
#    print('...............')

for i in range(len(img_paths)):
    frame = cv2.imread(img_paths[i])
    for box, pid in zip(boxess[i], per_idd[i]):
        cv2.rectangle(frame, (int(1920*box[0]), int(1080*box[1])),
                (int(1920*box[2]), int(1080*box[3])), (0, 255, 0), 2)
        cv2.putText(frame, pid, (int(1920*box[0]), int(1080*box[1])),
            cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2) 
    videoWriter.write(frame)
videoWriter.release()


#for i in range(len(img_path)):
#    frame = cv2.imread(img_path[i])
#
#    cv2.rectangle(frame, (int(1920*box[i][0]), int(1080*box[i][1])),
#            (int(1920*box[i][2]), int(1080*box[i][3])), (0, 255, 0), 2)
#    cv2.putText(frame, per_id[i], (int(1920*box[i][0]), int(1080*box[i][1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2) 
#    videoWriter.write(frame)
#videoWriter.release()

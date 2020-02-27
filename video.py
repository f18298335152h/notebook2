import cv2
import os
from copy import deepcopy
#fps = 2
#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(1920, 1080))
#with open('./frame_info.txt', 'r') as f:
#    for line in f:
#        img = line.strip()
#        frame = cv2.imread(img)
#        videoWriter.write(frame)
#videoWriter.release()

#img_id = []
img_path = []
per_id = []
boxs = []
fps = 4
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('saveVideo.avi',fourcc,fps,(1920, 1080))
with open('./frame_info.txt', 'r') as f:
    for line in f:
        arr = line.strip().split(' ')
#        img_id.append(arr[0])
        img_path.append(arr[1])
        box = arr[-1].split(',')
        boxs.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        per_id.append(box[4])

#frame = cv2.imread(img_path[200])
#cv2.rectangle(frame, (int(1920*boxs[200][0]), int(1080*boxs[200][1])),
#        (int(1920*boxs[200][2]), int(1080*boxs[200][3])), (0, 255, 0), 2)
#cv2.putText(frame, per_id[200], (int(1920*boxs[200][0]), int(1080*boxs[200][1])), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255,
#    0), 2)
#cv2.imwrite('test.jpg', frame)
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
#img_pp = list(set(img_p))
for i in  range(len(img_pp)):
    repeat.append([x for x in range(len(img_path)) if img_path[x] == img_pp[i]])
for ids in repeat:
    img_paths.append(img_path[ids[0]])
    tmp_b = []
    tmp_per = []
    for ind in ids:
#        print('ind = ', ind)
#        print(' box = ', boxs[ind])
#        print(' per id = ', per_id[ind])
        tmp_b.append(boxs[ind])
        tmp_per.append(per_id[ind])

    if len(tmp_b) !=0 and len(tmp_per) !=0:
        boxess.append(tmp_b)
        per_idd.append(tmp_per)
#    print('...............')


#print(len(img_paths[10]))
#print(len(per_idd[10]))
#print(len(boxess[10]))
#
#print(img_paths[10])
#print(per_idd[10])
#print(boxess[10])



#for val in img_path:
#    img_paths.append(val)
#    while True:
#        try:
#            last_index = img_path.index(val, last_index + 1)
#            boxess.append(boxs[last_index])
#            per_idd.append(per_id[last_index])
#        except ValueError:
#            break






for i in range(len(img_paths)):
    frame = cv2.imread(img_paths[i])
    for box, pid in zip(boxess[i], per_idd[i]):
        cv2.rectangle(frame, (int(1080*box[0]), int(1920*box[1])),
        (int(1080*box[2]), int(1920*box[3])), (0, 255, 0), 2)
        cv2.putText(frame, pid, (int(1080*box[0]), int(1920*box[1])),
            cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 0), 2)
    videoWriter.write(frame)
videoWriter.release()

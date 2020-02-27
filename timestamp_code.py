from copy import deepcopy
def timestamp_code():
    count = 0
    timestamplist = []
    img_path = []
    boxes = []
    with open('./frame_info.txt', 'r') as f:
        for line in f:
            imgpath_box_time = line.strip().split(' ')
            imgpath = imgpath_box_time[0]
            img_path.append(imgpath)
            box_time = imgpath_box_time[-1]
            box_time_list = box_time.split(',') 
            time = box_time_list[-1]
            box = box_time_list[0] +','+box_time_list[1]+','+box_time_list[2]+','+box_time_list[3]
            boxes.append(box)
            timestamplist.append(float(time))
    timestamplist_tmp = deepcopy(timestamplist) 
    timestamplist_end_ = []
    timestamplist_end = [-1]*len(timestamplist_tmp)
    for i  in range(len(timestamplist_tmp)):
        if timestamplist_tmp[i] not in timestamplist_end_:
            timestamplist_end_.append(timestamplist_tmp[i])
    #timestamplist_end = [-1]*len(timestamplist_tmp)
    for val in timestamplist_end_:
        for i in range(len(timestamplist_tmp)):
            if timestamplist_tmp[i] == val and val != -1:
                timestamplist_end[i] = count
            elif timestamplist_tmp[i] == val and val == -1:
                timestamplist_end[i] = -1
        count += 1
    with open('fam.lst', 'w') as f1:
        for i in range(len(img_path)):
            f1.write(img_path[i]+' '+boxes[i]+','+str(timestamplist_end[i])+'\n')

def compute_avg_frame():
    count = 0
    sum_time = 0
    with open('./frame.log', 'r') as f:
        for line in f:
            sum_time += float(line.strip())
            count += 1
    avg = sum_time / count
    print('avg = ', avg)


compute_avg_frame()

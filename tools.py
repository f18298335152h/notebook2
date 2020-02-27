import os
import ast
import cv2
import time
from PIL import Image
import numpy as np
import matplotlib
import torchvision.transforms.functional as F
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import Counter
from aitupu.interface.http_interface_detect import *
from aitupu.interface.http_interface_faceboxes import *
def read_file_vector(path):
    vec = []
    with open(path, 'r') as f:
        for line in f:
            data = line.strip()
            vec.append(float(data))
    print('len vec ',len(vec))

    return vec

def differ_tow_vector():
    ncnn = read_file_vector('/home/fenghui/tpncnn/ncnn.log')
    tpncnn = read_file_vector('/home/fenghui/tpncnn/tpncnn.log')

    for i in range(len(ncnn)):
        if ncnn[i] != tpncnn[i]:
            print('differ location ', i)
            return
    print('neon differ')


def plt_visibility_ced():
    threshold = np.arange(0, 1, 0.01)
    precision = np.random.rand(100)
    plt.plot(threshold, precision, label="test", color="red",linewidth=2)
    plt.xlabel("threshold")
    plt.ylabel("acc")
    plt.title("test result")
    plt.savefig("reslut.png")

def count_same():
    x = np.random.rand(100000000)
    y = np.random.rand(100000000)
    begin = time.time()
    val, count = np.unique(x, return_counts=True)
    end = time.time()
    cost = end - begin
    print('np cost = ', cost)

    begin1 = time.time()
    re = Counter(x)
    end1 = time.time()
    cost1 = end1 - begin1
    print('counter cost = ', cost1)


#def _server_detect_init():
#    detector = FaceDetectorIF()
#    return detector
#
#def _post_detect_server(batch_img_dict_detect,detector, detect_url):
#    bounding_box = []
#    res = detector.get_all_face_position(batch_img_dict_detect,1,detect_url,False)
#    for k,v in res.items():
#        x = int(v[0][0][0])
#        y = int(v[0][0][1])
#        width = int(v[0][0][2])
#        bounding_box.append((x,y,width))
#
#    return bounding_box
#
#def get_boundingbox():
#    DETECT_URL = ip_manager.get_server_url(ip_manager.SERVERID_RCNN_DETECT_FACE)
#    print('DETECT_URL = ', DETECT_URL)
#    detector = _server_detect_init()
#    badcase_lst = '/world/data-gpu-94/fenghui/mhg/face_age_badcase.lst'
#    img_list = []
#    BATCH_SIZE =50
#    with open(badcase_lst, 'r') as f:
#        for line in f:
#            img_list.append(line.strip())
#    btach_num = len(img_list) / BATCH_SIZE + 1
#    for bi in range(btach_num):
#        start = bi*BATCH_SIZE
#        end = min((bi+1)*BATCH_SIZE,len(img_list)-start)
#        img_batch = img_list[start:end]
#        for img in img_batch:
#            batch_img_dict_detect = {}
#            with open(img,'rb') as f:
#                batch_img_dict_detect[img] = f.read()
#                face_box = _post_detect_server(batch_img_dict_detect,detector,
#                        DETECT_URL)
#
#

#def get_boundingbox():
#    #url = 'http://172.25.52.70:23340/predict'
#    url = 'http://172.25.52.70:18000/ip/opt_env/face_ssd_mxnet_detect'
#    tupu_log.init_mlogger('http_interface_detect_test', '.', console=True)
#    detector = FaceDetectorIF()
#    badcase_lst = '/world/data-gpu-94/fenghui/mhg/face_age_badcase.lst'
#    img_list = []
#    with open(badcase_lst, 'r') as f:
#        for line in f:
#            img_list.append(line.strip())
#    image_dict = {}
#
#    for path in img_list:
#        with open(path, 'rb') as f:
#            image_dict[path] = f.read()
#    all_face = detector.get_all_face_position(image_dict, 1, url, is_crop=True)
#    print('len all face', len(all_face))

def get_boundingbox():
    from aitupu.common import tupu_log
    #from aitupu.interface.http_interface_faceboxes import *
    badcase_lst = '/world/data-gpu-94/fenghui/mhg/jimeiyou_neice.lst'
    badcase_box = '/world/data-gpu-94/fenghui/mhg/jimeiyou_neice_box.lst'
    img_list = []
    image_dict = {}
    url = 'http://172.26.3.68:12012'
    with open(badcase_lst, 'r') as f:
        for line in f:
            img_list.append(line.strip())
    for path in img_list:
        image_dict[path] = {}
        image_dict[path]['binary_image'] = cv2.imencode('.jpg', cv2.imread(path))[1]
    fb = Faceboxes()
    res = fb.inference(image_dict, url)
    
    with open(badcase_box, 'w') as f:
        f.write(json.dumps(res))


def crop_img():
    negative_case_list = '/world/data-gpu-94/fenghui/mks_vis_test/mks_badcase'
    negative_case_img = '/world/data-gpu-94/fenghui/mks_vis_test/detect_badcase'
    crop_path = '/world/data-gpu-94/fenghui/mks_vis_test/crop_badcase'
    with open(negative_case_list, 'r') as f:
        for line in f:
            arr = line.strip().split('.jpg')
            img_name = arr[0].split('/')[-1]
            arr[0] = arr[0] + '.jpg'
            tmp_dict = ast.literal_eval(arr[1].strip())
            xmin = tmp_dict['xmin']
            xmax = tmp_dict['xmax']
            ymin = tmp_dict['ymin']
            ymax = tmp_dict['ymax']
            xmin = int(xmin *0.8)
            xmax = int(xmax * 1.5)
            if xmax > 1024:
                xmax = 1024
            ymin = int(ymin * 0.8)
            ymax = int(ymax * 1.5)
            if ymax > 1024:
                ymax = 1024
            img = cv2.imread(arr[0])
            cropped = img[xmin:xmax, ymin:ymax]
            cropped_path = os.path.join(crop_path, img_name + '.jpg')
            print('cropped_path = ', cropped_path)
            print('cropped = ', cropped)
            cv2.imwrite(cropped_path, cropped)

def draw_rect_jmy():
    xmin = int(1920*0.1958333333)
    ymin = int(1080*0.1027777777)
    xmax = int(1920*0.31614583335)
    ymax = int(1080*0.31614583335)
    img = cv2.imread('/world/data-c22/test-data/bi/jimeiyou/xiaoxianli/0928_1018/wash/15703422559000.5281898022236065.jpg')
    print('img shape = ', img.shape)
    cv2.rectangle(img, (xmin, ymin),(xmax, ymax), (0, 255, 0), 2)
    cv2.imwrite('xxx.jpg', img)

def draw_rect_bad():
    img = cv2.imread('/world/data-c22/train-data/badcase/bi/face-age/08ef0220000a-38f9-9e11-9fac-07cfb6ef/15671510662320.802516149834505.jpg')
    xmin = int(727.8554077148438)
    ymin = int(311.8215637207031)
    xmax = int(792.2817993164062)
    ymax = int(403.585266)
    cv2.rectangle(img, (xmin, ymin),(xmax, ymax), (0, 255, 0), 2)
    cv2.imwrite('xxx.jpg', img)

def rename_img():
    path = '/world/data-gpu-94/fenghui/mks_vis_test/IBUG'
    img_list = os.listdir(path)
    for files in img_list:
        if ' ' in files:
            print(files)
#rename_img()
def copy_img_from_lst():
    files = '/world/data-gpu-94/fenghui/convert_hisi/resfacenet50_pruned_50_no_ibn_no_transpose_end/hisi_recognition_quantity_100.lst'
    dest = '/world/data-gpu-94/fenghui/quantized_img'
    with open(files, 'r') as f:
        for line in f:
            img_path = line.strip()
            img_name = img_path.split('/')[-1]
            os.system('cp %s %s'%(img_path, os.path.join(dest, img_name)))
#get_boundingbox()
#draw_rect_jmy()
copy_img_from_lst()

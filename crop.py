import torchvision.transforms.functional as F
from tpceph import tpread
import numpy as np
import cv2
import json
import os
import ast
import time
import random
mks = [[[0, 0]]*68]
vis = [[0]*68]
global imag_names
imag_names = []

json_data = []
with open('/world/data-gpu-94/fenghui/mks_vis_test/badcase_kouzhao_vet_train.json') as f:
    for line in f:
        arr = line.strip().split(' ')
        name = arr[0].split('/')[-1]
        imag_names.append(name)
def facemarks_bi_transform(img, roi_box, box_rect, img_path, is_face,
        crop_size, padding=-0.15):
    #save_dir = '/world/data-gpu-94/fenghui/mks_vis_test/badcase_big_pose_kouzhao_test'
    save_noface_dir = '/world/data-gpu-94/fenghui/class_face_end/origin/no_face'
    save_face_dir='/world/data-gpu-94/fenghui/class_face_end/origin/kouzhao_no_kouzhao_all'
    if is_face:
        save_dir = save_face_dir
    else:
        save_dir = save_noface_dir
    #img_name = img_path.split('/')[-1]
    img_name = str(time.time()) + '.jpg'
    save_img = os.path.join(save_dir, img_name)
    img = F.Image.fromarray(img)
    a = padding
    b = 1.0 + 2 * a
    w = roi_box[2]
    x = roi_box[0] - a * w
    x = max(0, x)
    y = roi_box[1] - a * w
    y = max(0, y)
    w = b * w
    img = F.crop(img, y, x, w, w)
    roi_box = np.float32([x, y, x+w, y+w])
    img = F.resize(img, (crop_size, crop_size))
    img = np.uint8(img)
    cv2.imwrite(save_img, img)
    if is_face:
        print('xxxx')
        with open('/world/data-gpu-94/fenghui/class_face_end/origin/kouzhao_no_kouzhao_all_crop.json', 'a') as f:
            f.write(save_img + ' 1'+'\n')
    else:
        with open('/world/data-gpu-94/fenghui/class_face_end/origin/no_face_crop.json', 'a') as f:
            f.write(save_img + ' 0'+'\n')
        
        #f.write(img_path+' ')
        #tmp_dict = {"facebox":[[box_rect[0][0], box_rect[0][1], box_rect[1][0], box_rect[1][1]]],'landmarks':mks, 'visibility':vis}
        ##tmp_dict = {'xmin':box_rect[0][0] * scale, 'ymin':box_rect[0][1] * scale, 'xmax':box_rect[1][0] * scale, 'ymax':box_rect[1][1] * scale,'landmarks':mks, 'visibility':vis}
        #json.dump(tmp_dict, f)
        #f.write('\n')
    
## -----------------  craw box on image box

#img = cv2.imread("/world/data-c24/train-data/mingchuang/face-detect/bad-quality/20200201-0211/15808833502570.12416840051287825.jpg")
#box = [[1197, 620], [1439, 862]] 
#xmin = box[0][0]
#ymin = box[0][1]
#xmax = box[1][0]
#ymax = box[1][1]
##xmin = box[0][0] * 1920
##ymin = box[0][1] * 1080
##xmax = box[2][0] * 1920
##ymax = box[2][1] * 1080
#x, y = xmin, ymin
#w = max(ymax - ymin, xmax - xmin)
#box = [int(x), int(y), int(w)]
#box_rect = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
#def rectangle(img, box_rect):
#    cv2.rectangle(img,(box_rect[0][0], box_rect[0][1]), (box_rect[1][0], box_rect[1][1]), (0, 255, 0), 2)
#    cv2.imwrite("rectxxxxx.jpg", img)
#rectangle(img, box_rect)

## ---------------- craw box on image end

def get_crop_img_from_badcase(data_list):
    img_path = []
    box = []
    with open(data_list) as f:
        for line in f:
            arr = line.split('\t')
            if int(arr[1]) != 4:
                continue
            img_path.append(arr[0])
            data_dict = eval(arr[3])
            box.append(data_dict['points'])
    print('box length : {} img_path length : {}'.format(len(box), len(img_path)))
    for i in range(len(box)):
        img = cv2.imread(img_path[i])
        xmin = box[i][0][0] * 1920
        ymin = box[i][0][1] * 1080
        xmax = box[i][2][0] * 1920
        ymax = box[i][2][1] * 1080
        x, y = xmin, ymin
        w = max(ymax - ymin, xmax - xmin)
        box_ = [int(x), int(y), int(w)]
        box_rect = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
        try:
            facemarks_bi_transform(img, box_, box_rect, img_path[i], 450)
        except:
            continue
#get_crop_img_from_badcase("/world/data-gpu-94/fenghui/mks_vis_test/badcase_kouzhao_all.json")

def revise_mks(json_list):
    with open(json_list, 'r') as f:
        for line in f:
            arr = line.strip().split('jpg')
            arr[0] = arr[0] + 'jpg'
            tmp_dict = ast.literal_eval(arr[1].strip())
            tmp_dict_ = {'xmin': tmp_dict['xmin'], 'ymin':tmp_dict['ymin'],
                    'xmax': tmp_dict['xmax'], 'ymax': tmp_dict['ymax'],
                    'landmarks':mks, 'visibility':tmp_dict['visibility']}
            with open('badcase_kouzhao_vet.json', 'a') as f1:
                f1.write(arr[0]+' ')
                json.dump(tmp_dict_, f1)
                f1.write('\n')
#revise_mks('/world/data-gpu-94/fenghui/mks_vis_test/badcase_kouzhao.json')
def revise_classification(json_list):
    count = 0
    with open(json_list, 'r') as f:
        for line in f:
            arr = line.strip().split('\t')
            if count > 15000:
                with open('face_kouzhao_train.json', 'a') as f1:
                    f1.write(arr[0]+' ' + '1'+'\n')
            else:
                with open('face_kouzhao_test.json', 'a') as f2:
                    f2.write(arr[0]+' ' + '1'+'\n')
            count += 1
#revise_classification('/world/data-c9/dl-data/5a3b7fda1c9da69b2630e71a/5e4a608231493d4a71028700/15819326740450.9062070174475889_list.json')

def revise_classification_face_nokouzhao(json_list):
    count = 0
    with open(json_list, 'r') as f:
        for line in f:
            arr = line.strip().split(' ')
            if count > 100000:
                with open('face_no_kouzhao_train.json', 'a') as f1:
                    f1.write(arr[0]+' ' + '2'+'\n')
            else:
                with open('face_no_kouzhao_test.json', 'a') as f2:
                    f2.write(arr[0]+' ' + '2'+'\n')
            count += 1
#revise_classification_face_nokouzhao('/world/data-gpu-94/fenghui/face_classification/bi_20200210_mask.lst')


def tpceph_image(json_list):
    save_noface_dir = '/world/data-gpu-94/fenghui/face_classification/no_face'
    #save_face_dir = '/world/data-gpu-94/fenghui/face_classification/face'

    with open(json_list, 'r') as f:
        for line in f:
            arr = line.strip().split('\t')
            img_path = arr[0]
            #img_name = img_path.split('/')[-1]
            image = tpread(img_path)
            try:
                infor = ast.literal_eval(arr[1])
            except:
                continue
            for inf in infor:
                xmax = inf['xmax']
                ymax = inf['ymax']
                xmin = inf['xmin']
                ymin = inf['ymin']
                #xmax = 0.5*inf['xmax']
                #ymax = 0.5*inf['ymax']
                #xmin = 0.5*inf['xmin']
                #ymin = 0.5*inf['ymin']
                name = inf['name']
                x, y = xmin, ymin
                w = max(ymax - ymin, xmax - xmin)
                box_ = [int(x), int(y), int(w)]
                box_rect = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
                #### face
                #if name == '0':
                #    try:
                #        facemarks_bi_transform(image, box_, box_rect, img_path,
                #             True, 64)
                #    except:
                #        continue
                #### no face
                if name == '1':
                    try:
                        facemarks_bi_transform(image, box_, box_rect, img_path,
                             False, 64)
                    except:
                        continue
#tpceph_image('/world/data-gpu-94/fenghui/class_face_end/origin/no_face.json')


def kouzhao_face(json_list):
    count = 0
    with open(json_list, 'r') as f:
        for line in f:
            arr = line.strip().split('\t')
            img_path = arr[0]
            image = cv2.imread(img_path)
            tmp_dict = ast.literal_eval(arr[2].strip())
            box = tmp_dict['points']
            xmin = box[0][0] * 1920
            ymin = box[0][1] * 1080
            xmax = box[1][0] * 1920
            ymax = box[1][1] * 1080
            x, y = xmin, ymin
            w = max(ymax - ymin, xmax - xmin)
            box_ = [int(x), int(y), int(w)]
            box_rect = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
            try:
                facemarks_bi_transform(image, box_, box_rect, img_path,
                     True, 64)
                count += 1
                print('count = ', count)
            except:
                continue
#kouzhao_face('/world/data-gpu-94/fenghui/class_face_end/origin/kouzhao_no_kouzhao_all.json')

def test_imag_box():
    img = cv2.imread("/world/data-c24/train-data/mingchuang/face-detect/mask/20200214-0220/xiaoxianli/15820950834110.5662120390317533.jpg")
    #box = [[0.04479166666666667,0.5037037037037037],[0.15,0.5037037037037037],[0.15,0.6907407407407408],[0.04479166666666667,0.6907407407407408]]
    #box = []
    box = [[0.37083333333333335,0.08425925925925926],[0.4765625,0.2722222222222222]]
    #xmin = box[0][0]
    #ymin = box[0][1]
    #xmax = box[1][0]
    #ymax = box[1][1]
    xmin = box[0][0] * 1920
    ymin = box[0][1] * 1080
    xmax = box[1][0] * 1920
    ymax = box[1][1] * 1080
    x, y = xmin, ymin
    w = max(ymax - ymin, xmax - xmin)
    box = [int(x), int(y), int(w)]
    box_rect = [[int(xmin), int(ymin)], [int(xmax), int(ymax)]]
    cv2.rectangle(img,(box_rect[0][0], box_rect[0][1]), (box_rect[1][0], box_rect[1][1]), (0, 255, 0), 2)
    cv2.imwrite("yyyy.jpg", img)



def get_test_data_from_mask(json_list):
    with open(json_list, 'r') as f:
        for line in f:
            arr = line.strip().split(' ')
            with open('tmp_succ.lst', 'a') as f1:
                f1.write(arr[0]+' 2'+'\n')
#get_test_data_from_mask('/world/data-gpu-94/fenghui/face_classification/tmp.lst')

def get_detect_bahcase_aug(json_list):
    save_dir = '/world/data-gpu-94/fenghui/classification/no_face3_valid'
    with open(json_list, 'r') as f:
        for line in f:
            img_path = line.strip().split(' ')[0]
            img = cv2.imread(img_path,1)
            imgInfo = img.shape
            h = imgInfo[0]
            w = imgInfo[1]
            p = random.random()
            if p > 0.4:
                matRot = cv2.getRotationMatrix2D((h*0.5, w*0.5), 30, 0.5)
            else:
                matRot = cv2.getRotationMatrix2D((h*0.5, w*0.5), 70, 0.5)
            dst = cv2.warpAffine(img, matRot, (h, w))
            name = str(time.time())
            if '.jpg' not in name:
                name += '.jpg'
            save_img = os.path.join(save_dir, name)
            cv2.imwrite(save_img, dst)
            #cv2.imwrite('/world/data-gpu-94/fenghui/classification/kouzhao_face_clear_aug_test-20/'+name,dst)
            with open('no_face_aug_valid.json', 'a') as f1:
                f1.write(save_img +' 0'+'\n')
                
#get_detect_bahcase_aug('/world/data-gpu-94/fenghui/classification/no_face_data_3_end.json')


def clear_kouzhao(json_list):
    save_dir = '/world/data-gpu-94/fenghui/classification/kouzhao_face_clear'
    with open(json_list, 'r') as f:
        for line in f:
            img_path = line.strip().split(' ')[0]
            img = cv2.imread(img_path)
            name = img_path.split('/')[-1] 
            save_img = os.path.join(save_dir, name)
            cv2.imwrite(save_img, img)
            with open('face_kouzhao_data_train_shuffle_clear.json', 'a') as f1:
                f1.write(save_img +' 1'+'\n')
#clear_kouzhao('/world/data-gpu-94/fenghui/classification/face_kouzhao_data_train_shuffle.json')




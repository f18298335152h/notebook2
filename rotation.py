import os
import cv2
import random
import time
def start_rotation(json_list):
    save_dir = '/world/data-gpu-94/fenghui/class_face_end/origin/no_face_aug_rota_gauss'
    with open(json_list, 'r') as f:
        for line in f:
            img_path = line.strip().split(' ')[0]
            img = cv2.imread(img_path,1)
            imgInfo = img.shape
            h = imgInfo[0]
            w = imgInfo[1]
            p = random.random()
            
            ###flip
            #if p > 0.4 and p < 0.8:
            #    img = cv2.flip(img, 1)
            #    #matRot = cv2.getRotationMatrix2D((h*0.5, w*0.5), 30, 1)
            #elif p < 0.4:
            #    img = cv2.flip(img, 0)
            #else:
            #    img = cv2.flip(img, -1)
            #    #matRot = cv2.getRotationMatrix2D((h*0.5, w*0.5), 70, 1)
            #dst = cv2.warpAffine(img, matRot, (h, w))
            
            ### rotation
            if p > 0.5:
                matRot = cv2.getRotationMatrix2D((h*0.5, w*0.5), 15, 1)
            else:
                matRot = cv2.getRotationMatrix2D((h*0.5, w*0.5), 37, 1)
            dst = cv2.warpAffine(img, matRot, (h, w))
            dst = cv2.GaussianBlur(dst, (7,7), 1.5)

            name = str(time.time())
            if '.jpg' not in name:
                name += '.jpg'
            save_img = os.path.join(save_dir, name)
            #cv2.imwrite(save_img, img)
            cv2.imwrite(save_img, dst)
            #cv2.imwrite('/world/data-gpu-94/fenghui/classification/kouzhao_face_clear_aug_test-20/'+name,dst)
            with open('/world/data-gpu-94/fenghui/class_face_end/origin/no_face_data_aug_rota_gauss.json', 'a') as f1:
                f1.write(save_img +' 0'+'\n')
#
start_rotation('/world/data-gpu-94/fenghui/class_face_end/origin/no_face_data_aug.json')

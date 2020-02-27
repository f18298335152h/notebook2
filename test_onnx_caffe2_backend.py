import cv2
import onnx
import numpy as np
import caffe2.python.onnx.backend as backend
def check_onn_MobileHG():
    model = onnx.load("./MobileHG_pose_8.onnx")
    predictor = backend.prepare(model, device="CUDA")
    x = cv2.imread('/world/data-gpu-94/fenghui/onnx_model/wyq_128_128.jpg')
    img = x.copy()
    batch = []
    x = cv2.resize(x, (128, 128))
    x = x / 256.0
    x = x.transpose(2, 0, 1)
    x = np.float32([x])
    for i in range(8):
        batch.append(x)
    batch = np.asarray(batch)
    batch = batch.reshape(8, 3, 128, 128)
    
    res = predictor.run(batch)
    print(res)


def check_onn_recognition():
    model = onnx.load("/world/data-gpu-94/fenghui/mobilefacenet_3.onnx")
    predictor = backend.prepare(model, device="CUDA")
    x = cv2.imread('/world/data-gpu-94/fenghui/onnx_model/wyq_128_128.jpg')
    img = x.copy()
    batch = []
    x = cv2.resize(x, (112, 112))
    x = x / 256.0
    x = x.transpose(2, 0, 1)
    x = np.float32([x])
    res = predictor.run(x)
    print(res)

check_onn_recognition()

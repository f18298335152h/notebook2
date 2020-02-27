from models.mobilenet_v2 import MobileHG_light
from models.mobilenet_v2_dpdm_1_1 import  MobileNetV2_light
import torch
import cv2
import numpy as np
import onnx

def export_onnx():
    x = torch.randn(1, 3 ,96, 96)
    x = cv2.imread('/world/data-gpu-94/fenghui/onnx_model/wyq_128_128.jpg')
    x = cv2.resize(x, (96, 96))
    x = x / 255.0
    x = x.transpose(2, 0, 1)
    x = np.float32([x])
    x = torch.from_numpy(x)

    model = MobileNetV2_light()
    #model = DPDM()
    model.eval()
    ckpt = '/world/data-gpu-94/fenghui/mhg/test_68_300wlp_2.5D_mbv2_2_2_vis_136_prob_1_1/vis/mhg/debug_metric_false/MobileNetV2_light_best_model.pth'
    ckpt_dpdm = '/world/data-gpu-94/fenghui/dpdm/DPDM_0111_11:04:19_epoch_0420.pth'
#    #torch.save(model.state_dict(), 'ShuffleNet_V1_best_model.pth')
    state_dict_mobile = torch.load(ckpt)

    state_dict_dpdm = torch.load(ckpt_dpdm)
    #print('type state_dict_mobile = ', type(state_dict_mobile))
    #print('state_dict_mobile keys = ', state_dict_mobile.keys())
    #print('state_dict_dpdm keys = ', state_dict_dpdm.keys())
    state_dict_dpdm = {'dpdm.'+k:state_dict_dpdm[k] for k in state_dict_dpdm.keys()}
    
    state_dict = dict(state_dict_mobile.items() + state_dict_dpdm.items())
    #print('state_dict . keys = ', state_dict.keys())
    #print(' state_dict[dpdm.w] = ',state_dict['dpdm.w'])
    #print('state_dict key', state_dict.keys())
    #state_dict = {k:state_dict[k] for k in state_dict.keys() if 'features.15' or 'fc_is_mks' not in k}
    
#    some_dict = {k[0:11]+'.conv'+k[11:]:state_dict[k] for k in state_dict.keys() if 'features.15' in k} 
#    state_dict = dict(state_dict.items() + some_dict.items())
    model.load_state_dict(state_dict)
    mks = model(x)
    #model.load_state_dict(state_dict)
#    torch.onnx.export(model, x, 'MobileNetV2_96_vis_fc_dpdm_no_dr.onnx', export_params=True, verbose=True)
    #torch.onnx.export(model, x, 'MobileNetV2_96_512_del.onnx', export_params=True, verbose=True)

def check_onnx():
    model = onnx.load("./MobileNetV2_96_512_dpdm.onnx")

    #print('model = ', model['dpdm.w'])
    #onnx.checker.check_model(model)
    #print(onnx.helper.printable_graph(model.graph))
    print(model.graph.input)
    print(model.graph.initializer)
if __name__=='__main__':
    export_onnx()
    #check_onnx()

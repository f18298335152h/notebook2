from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
tpncnn = []
ncnn = []


def read_file(path):
    ncnn = []
    with open(path, 'r') as f:
        for line in f:
            ncnn.append(float(line.strip()))
    return ncnn
def cos_similarity():
    vec1 = read_file('/home/fenghui/tpncnn/examples/examples/594.log') 
    vec2 = read_file('/world/data-gpu-94/fenghui/tpncnn/examples/examples/594.log')
    lii = []
    lii.append(vec1)
    lii.append(vec2)
    rec1 = cosine_similarity(lii)
    print('cos similarity = ', rec1)

def l2_dis():
    vec1 = read_file('/home/fenghui/tpncnn/examples/examples/594.log')
    vec2 = read_file('/world/data-gpu-94/fenghui/tpncnn/examples/examples/594.log')
    tpncnn = np.array(vec1)
    ncnn = np.array(vec2)
    num=float(np.sum(tpncnn*ncnn))
    denom=np.linalg.norm(tpncnn)*np.linalg.norm(ncnn)




if __name__ == '__main__':
    cos_similarity()
    l2_dis()

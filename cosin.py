import numpy as np
from scipy.spatial.distance import cosine
from scipy import spatial
def get_matrix():
    matrix = []
    with open('./fea1.txt', 'r') as f:
        for line in f:
            arr = line.strip().split(' ')
            for val in arr:
                if val == '':
                    continue
                print(val)


def cos():
    matrix = []
    fea = []
    count = 0
    with open('./matrix_clean.txt', 'r') as f:
        matrix_tmp1 = []
        matrix_tmp2 = []
        matrix_tmp3 = []
        for line in f:
            if line.strip() == '':
                count += 1
                continue
            if count < 512:
                matrix_tmp1.append(float(line.strip()))
            elif count > 512 and count < 1025 :
                matrix_tmp2.append(float(line.strip()))
            else:
                matrix_tmp3.append(float(line.strip()))
            count += 1
        matrix.append(matrix_tmp1)
        matrix.append(matrix_tmp2)
        matrix.append(matrix_tmp3)
    with open('./feature.txt', 'r') as f1:
        for line in f1:
            fea.append(float(line.strip()))
    print('fea -1 = ', fea[-1])
    print('0 0 matrix = ',matrix[0][0])
    print('0 511 matrix = ',matrix[0][511])
    print('1 0 matrix = ',matrix[1][0])
    print('1 511 matrix = ',matrix[1][511])
    print('2 0 matrix = ',matrix[2][0])
    print('2 511 matrix = ',matrix[2][511])
    print(' fea norm = ', np.linalg.norm(fea)) 
    print(' matrix norm = ', np.linalg.norm(matrix))
    for mat in matrix:
        cos = np.dot(mat,fea)/(np.linalg.norm(mat)*(np.linalg.norm(fea)))
        print(' mat norm = ', np.linalg.norm(mat))
        sim = 1 - spatial.distance.cosine(mat, fea)
        print('cos = ', cos)
        print(' sim = ', sim)


def get_two_list():
    fea1 = []
    fea2 = []
    with open('/home/fenghui/Inst_models_convert/378.log', 'r') as f:
        for line in f:
            fea1.append(float(line.strip()))
    with open('/world/data-gpu-94/fenghui/feature/378.log', 'r') as f:
        for line in f:
            fea2.append(float(line.strip()))
    fea1 = np.mat(fea1)
    fea2 = np.mat(fea2)
    a = fea1
    b = fea2
    num = float(fea2 * fea1.T)
    denom = np.linalg.norm(fea1) * np.linalg.norm(fea2)
    cos = num / denom
    print('cos = ', cos)

get_two_list()

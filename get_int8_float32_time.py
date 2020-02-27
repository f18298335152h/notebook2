import os
import sys



def get_int8_float32_file(float32, int8):
    result1 = []
    result2 = []
    result = []
    with open(float32) as f1:
        for line in f1:
            result1.append(line.strip())

    with open(int8) as f2:
        for line in f2:
            result2.append(line.strip())

    for i in range(len(result1)):
        arr_1 = result1[i].split('|')
        arr_2 = result2[i].split('|')
        arr_1_str = '|'
        for j in range(len(arr_1)):
            if arr_1[j] == '':
                continue
            if j == 2:
                continue
            if j == 3:
                arr_1_str = arr_1_str  +  arr_1[j] + '|' + arr_2[j]
            arr_1_str  = arr_1_str + arr_1[j] + '|'
        print(arr_1_str) 

def statics_int8_float32_conv_time(float32, int8):
    result1 = {}
    result2 = {}
    result = {}
    with open(float32) as f1:
        for line in f1:
            arr_float = line.strip().split('\t')
            if arr_float[1] in result1.keys():
                result1[arr_float[1]].append([arr_float[0], float(arr_float[2])])
            else:
                result1[arr_float[1]] = []
    with open(int8) as f2:
        for line in f2:
            arr_int8 = line.strip().split('\t')
            if arr_int8[1] in result2.keys():
                result2[arr_int8[1]].append([arr_int8[0], float(arr_int8[2])])
            else:
                result2[arr_int8[1]] = []
    float32_dic = {}
    for k, vals in result1.items():
        sum_f = 0
        for val in vals:
            sum_f += val[1]
        avg = sum_f / len(vals) 
        float32_dic[k] = [avg, vals[0][0]]
    int8_dic = {}
    for k, vals in result2.items():
        sum_conv = 0
        sum_quanti = 0
        sum_dequanti = 0
        count_conv = 0
        count_quanti = 0
        count_dequanti = 0
        conv= ''
        for val in vals:
            if 'quantize' == val[0]:
                sum_quanti += val[1]
                count_quanti += 1
            elif 'dequantize' == val[0]:
                sum_dequanti += val[1]
                count_dequanti += 1
            else:
                conv = val[0]
                sum_conv += val[1]
                count_conv += 1
        if count_quanti == 0 or count_dequanti == 0 or count_conv == 0:
            continue
        avg_conv = sum_conv /count_conv
        avg_quanti = sum_quanti / count_quanti
        avg_dequanti = sum_dequanti / count_dequanti
        int8_dic[k] = {'quantize':avg_quanti,'dequantize':avg_dequanti,conv:avg_conv} 
    for ky, vls in float32_dic.items():
        res = '|'
        res += ky
        for vl in vls:
            res += '|' + str(vl)
        res += '|'
        print(res)

    print('\n\n\n')
    print('int8\n')
    print('int8_dic = ',len(int8_dic.keys()))
    for ks, vls in int8_dic.items():
        res = '|' + ks + '|'
        for k, v in vls.items():
            res += str(v) 
            res += '|'
            if k != 'quantize' and  k!= 'dequantize':
                sstr = k;
        print(res + sstr + '|')

if __name__=='__main__':
    float32 = sys.argv[1]
    int8 = sys.argv[2]
    statics_int8_float32_conv_time(float32, int8)
    #get_int8_float32_file(float32, int8)

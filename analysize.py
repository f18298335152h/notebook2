import re
import sys
import os

def get_best_cost_time_file(cost_time_dir):
    '''

        return the best thread file 

    Args:
        cost_time_dir: string  folder of different thread cost time file

    Return:
        the best cost time file and avg time
    '''
    if not os.path.exists(cost_time_dir):
        print ' {0} is not exists !'.format(cost_time_dir)
    avg_time_list = []
    for cost_file in os.listdir(cost_time_dir):
        cost_file = os.path.join(cost_time_dir, cost_file)
        with open(cost_file,'r') as f:
            f.seek(-44,2)
            avg_time  = f.readline()
            avg_time = float(avg_time.split('=')[-1].strip())
            avg_time_list.append((cost_file, float(avg_time)))
    avg_time_sorted_list = sorted(avg_time_list, key=lambda tup: tup[1])
    
    return avg_time_sorted_list[0]

def get_every_layer(best_file_tuple):
    '''
        re to match the every layer (operator type,layer name,cost time,feature map) 
    
    Arsg:
        best_file_tuple: tuple (best file path,cost time )
    Return
        [(operator,layer_name,cost_time,feature_map)]
    '''
    every_layer_list = []
    layer_name_list = []
    best_file = best_file_tuple[0]
    cost_time = best_file_tuple[1]
    best_thread = best_file.split('/')[1].split('.')[0]

#    layer_record = re.compile(r'^(.*?)\s+(.*?)\s+(.*?ms).*?feature_map:\s+(.*?)\s+inch:\s+(.*?)\s+outch:\s+(.*?)$')
#    layer_record_ = re.compile('r^(.*?)\s+(.*?)\s+(.*?ms).*?|\s+$')
    index = 0
    with open(best_file,'r') as f:
        for line in f:
            index += 1
            if index < 7:
                continue
            arr = line.strip().split('\t')
            tt = []
            for op in arr:
                cl_op = op.strip()
                if cl_op == '|':
                    continue
                tt.append(cl_op)
            every_layer_list.append(tt)
    del every_layer_list[-1]
#    print every_layer_list
    for every_layer in every_layer_list:
        if every_layer[1] in layer_name_list:
            continue
        layer_name_list.append(every_layer[1])

    return every_layer_list, cost_time, best_thread,layer_name_list
def cal_avg_time_layer(every_layer_list, best_time, best_thread, layer_name_list):
    '''
        cal avg time for every the same layer_name
    
    Args:
        every_layer_list: [(operator,layer_name,cost_time,feature_map)]


    '''
    layer_operator_dict = {}
    layer_avg_time_dict = {}
    for every_layer in every_layer_list:
        if every_layer[1] not in layer_operator_dict.keys():
            layer_operator_dict[every_layer[1]] = []
        if len(every_layer) == 3:
            layer_operator_dict[every_layer[1]].append((every_layer[0],every_layer[2].replace('ms','')))
        elif len(every_layer) == 6:
            layer_operator_dict[every_layer[1]].append((every_layer[0],every_layer[2].replace('ms',''),every_layer[3],every_layer[4],every_layer[5]))
        elif len(every_layer) == 8:
            layer_operator_dict[every_layer[1]].append((every_layer[0],every_layer[2].replace('ms',''),every_layer[3],every_layer[4],every_layer[5],every_layer[6],every_layer[7]))
   
    for layer_name, same_layer_attr in layer_operator_dict.items():
        sum_time = 0.0
        for layer_attr in same_layer_attr:
            sum_time += float(layer_attr[1])
        layer_avg_time = round(sum_time / len(same_layer_attr),4)
        layer_avg_time_dict[layer_name] = []
        for idx, attr in enumerate(same_layer_attr[0]):
            layer_avg_time_dict[layer_name].append(attr)
        layer_avg_time_dict[layer_name].append(layer_avg_time)

    for name in layer_name_list:
        if name in layer_avg_time_dict.keys():
            if len(layer_avg_time_dict[name]) == 3:
                print '|'+str(layer_avg_time_dict[name][0])+'|'+str(name)+'|'+str(layer_avg_time_dict[name][2])+'|'
            elif len(layer_avg_time_dict[name]) == 6:
                print '|'+str(layer_avg_time_dict[name][0])+'|'+str(name)+'|'+\
                    str(layer_avg_time_dict[name][5])+'|'+str(layer_avg_time_dict[name][2])+\
                    '|'+str(layer_avg_time_dict[name][3])+'|'+str(layer_avg_time_dict[name][4])+'|'
            elif len(layer_avg_time_dict[name]) == 8:
                print '|'+str(layer_avg_time_dict[name][0])+'|'+str(name)+'|'+\
                    str(layer_avg_time_dict[name][7])+'|'+str(layer_avg_time_dict[name][2])+\
                    '|'+str(layer_avg_time_dict[name][3])+'|'+str(layer_avg_time_dict[name][4])+\
                    '|'+str(layer_avg_time_dict[name][5])+'|'+str(layer_avg_time_dict[name][6])+'|'

    '''
    for layer_name,operator_avg_time in sorted_layer_avg_time_dict.items():
        if len(operator_avg_time) == 7:
            print '|'+str(operator_avg_time[0])+'|'+str(layer_name)+'|'+\
                str(operator_avg_time[1])+'|'+str(operator_avg_time[2])+\
                '|'+str(operator_avg_time[3])+'|'+str(operator_avg_time[4])+\
                '|'+str(operator_avg_time[5])+'|'+str(operator_avg_time[6])+'|'
        else:
            print '|'+str(operator_avg_time[0])+'|'+str(layer_name)+'|'+\
            str(operator_avg_time[1])+'|'+str(operator_avg_time[2])+\
            '|'+str(operator_avg_time[3])+'|'+str(operator_avg_time[4])+'|'
#        print 'layer_name = ',layer_name
#        print 'operator = ',operator_avg_time[0]
#        print 'avg_time = ',operator_avg_time[1]
    '''
    print(' layer count = ',len(layer_avg_time_dict.keys()))
    print(' thread = ',best_thread)

if __name__=='__main__':
    cost_time_dir = sys.argv[1]
    best_file_tuple = get_best_cost_time_file(cost_time_dir)
    every_layer_list, best_time, best_thread, layer_name_list = get_every_layer(best_file_tuple)
    cal_avg_time_layer(every_layer_list, best_time, best_thread, layer_name_list)

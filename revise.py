
def revis_file(json_list):
    with open(json_list, 'r') as f:
        for line in f:
            arr = line.strip().split(' ')
            with open('face_no_ouzhao_data_valid_.json','a') as f:
                f.write(arr[0]+' '+'2'+'\n')
revis_file('/world/data-gpu-94/fenghui/classification/face_no_kouzhao_data_valid.json') 

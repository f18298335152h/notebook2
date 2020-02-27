import os
#f_dir = '/Users/mac/PycharmProjects/ncnn_bak/int8-float32-model'
#f_dir = '/Users/mac/PycharmProjects/ncnn_bak/squeeze_model/'
f_dir = '/Users/mac/PycharmProjects/ncnn_bak/mfn_noSE'
list_file = []
list_file = os.listdir(f_dir)
list_file_middle = []
for file_ in list_file:
    if file_ in list_file_middle:
        continue
    file_name = file_.split('.')[0]
    params = file_name +'.param'
    bins = file_name +'.bin'
    list_file_middle.append(params)
    list_file_middle.append(bins);
    os.system('mkdir %s'%(file_name))
    log_name = file_name+'/'+file_name+'.log'
    os.system('adb shell ./data/local/build-android-armv7/benchmark/benchncnn 100 6 0 -1 %s %s  >&  %s' %(bins, params, log_name))

#!/usr/bin/env python
# coding=utf-8
import os
label_file = open('test_label.txt','w')
for maindir, subdir, file_name_list in os.walk('casia_clean_test'):
    for filename in file_name_list:
        apath = os.path.join(maindir, filename)
        label_file.write(apath+' '+maindir.split('/')[-1]+'\n')
        
label_file.close()

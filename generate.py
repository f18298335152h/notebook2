


x = ['a', 'b', 'c', 'd']

with open('input_32.txt','w') as f:
    for i in range(1*1048576+10):
        for val in x:
            f.write(val)

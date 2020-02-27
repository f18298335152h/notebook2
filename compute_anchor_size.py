import math
# feature maps layers
mbox_source_layers = ['32', '16', '8']

# s_min s_max
min_ratio = 20
max_ratio = 90
min_dim = 1024
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in range(min_ratio, max_ratio + 1, step):
    min_sizes.append(min_dim * ratio / 100.)
    max_sizes.append(min_dim * (ratio + step) / 100.)

min_sizes = [min_dim * 10  / 300.] + min_sizes
max_sizes = [min_dim * 20  / 300.] + max_sizes

print('min_sizes = ', min_sizes)
print('max_sizes = ', max_sizes)

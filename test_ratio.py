import numpy as np 

Lamda = 0.8 
prune_n = 4
prune_m = 8

all_layer_ratio = [4.264982747290418, 4.1755982631228745, 4.723070569606643, 5.2538333161507245, 4.682753370215856, 4.490671009597383, 4.644753648827113, 4.511491745864789, 4.309698707699158, 4.251974728440992, 4.155141701970075, 4.2943801286924685, 4.19993474693496, 4.154526507916228, 4.068789210344226, 3.9009415423931855, 3.779056529306995, 3.7748494914158637, 3.8049070328628463, 3.727245330810547, 3.640476780234223, 3.730023339622379, 3.6323221236312944, 3.710374189782019, 3.6196975510355105, 3.7600744573563496, 3.6308575170645443, 3.575712905646606, 3.5566537491398154, 3.4064574562823835, 3.418059176113939, 3.0952621618083107]

# step 4: adjust the outlier ratio by z-scaling
print("before adjustment", all_layer_ratio)

all_layer_ratio = np.array(all_layer_ratio)
all_layer_ratio = (all_layer_ratio - all_layer_ratio.min()) * (
    1
    / (all_layer_ratio.max() - all_layer_ratio.min())
    * Lamda  # 0.08 by default
)
all_layer_ratio = all_layer_ratio - np.mean(all_layer_ratio)

all_layer_ratio = np.round(all_layer_ratio)

# Here is the key step to adjust the outlier ratio with N:M
for i in range(len(all_layer_ratio)):
    if all_layer_ratio[i] == 1.0:
        all_layer_ratio[i] = 2.0

all_layer_ratio = prune_n - all_layer_ratio

print("after adjustment", all_layer_ratio)
import random

import imageio

dataset_csv = '/home/gaoha/epe/code/Carla/OverfitTest/overfit-carla2cs-filtered.csv'

# Open dataset_csv
with open(dataset_csv, 'r') as f:
    lines = f.readlines()[1:]

    # Randomly shuffle the lines
    random.shuffle(lines)

    # 'src_path', 'src_r0', 'src_r1', 'src_c0', 'src_c1', 'dst_path', 'dst_r0', 'dst_r1', 'dst_c0', 'dst_c1'
    for k in range(20):

        # First, generate a random hash value
        hash_val = ''.join([random.choice('0123456789abcdef') for _ in range(6)])

        tmp = lines[k].strip().split(',')

        src_r0 = int(tmp[1])
        src_r1 = int(tmp[2])
        src_c0 = int(tmp[3])
        src_c1 = int(tmp[4])
        dst_r0 = int(tmp[6])
        dst_r1 = int(tmp[7])
        dst_c0 = int(tmp[8])
        dst_c1 = int(tmp[9])

        # Open src_path as img
        img = imageio.imread(tmp[0])
        img2 = imageio.imread(tmp[5])
        imageio.imwrite(f'/home/gaoha/epe/Carla/sample3/{hash_val}_Carla.png', img[src_r0:src_r1, src_c0:src_c1, :])
        imageio.imwrite(f'/home/gaoha/epe/Carla/sample3/{hash_val}_CityScapes.png', img2[dst_r0:dst_r1, dst_c0:dst_c1, :])



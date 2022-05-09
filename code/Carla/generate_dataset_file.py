import glob
import os
import argparse

import IPython
import imageio

##############################################################
# Constants
##############################################################

# CityScape Labels
from pathlib import Path

import numpy as np
import torch


class CityScape:
    def __init__(self):
        self.name2color = {
            'unlabeled': (0, 0, 0),
            'road': (128, 64, 128),
            'sidewalk': (244, 35, 232),
            'building': (70, 70, 70),
            'wall': (102, 102, 156),
            'fence': (190, 153, 153),
            'pole': (153, 153, 153),
            'traffic_light': (250, 170, 30),
            'traffic_sign': (220, 220, 0),
            'vegetation': (107, 142, 35),
            'terrain': (152, 251, 152),
            'sky': (70, 130, 180),
            'person': (220, 20, 60),
            'rider': (255, 0, 0),
            'car': (0, 0, 142),
            'truck': (0, 0, 70),
            'bus': (0, 60, 100),
            'train': (0, 80, 100),
            'motorcycle': (0, 0, 230),
            'bicycle': (119, 11, 32),
            'rail_track': (230, 150, 140),
            'ground': (81, 0, 81),
            'guard_rail': (180, 165, 180),
            'bridge': (150, 100, 100),
        }

        self.color2name = {v: k for k, v in self.name2color.items()}

# Carla Labels
class Carla:
    def __init__(self):
        self.name2color = {
            'unlabeled': (0, 0, 0), ###
            'building': (70, 70, 70), ###
            'fence': (100, 40, 40), ###
            'other': (55, 90, 80),
            'person': (220, 20, 60), ###
            'pole': (153, 153, 153), ###
            'road_line': (157, 234, 50),
            'road': (128, 64, 128), ###
            'sidewalk': (244, 35, 232), ###
            'vegetation': (107, 142, 35), ###
            'car': (0, 0, 142), ###
            'wall': (102, 102, 156), ###
            'traffic_sign': (220, 220, 0), ###
            'sky': (70, 130, 180), ###
            'ground': (81, 0, 81), ###
            'bridge': (150, 100, 100), ###
            'rail_track': (230, 150, 140), ###
            'guard_rail': (180, 165, 180), ###
            'traffic_light': (250, 170, 30), ###
            'static': (110, 190, 160), ###
            'dynamic': (170, 120, 50),
            'water': (45, 60, 150), ###
            'terrain': (145, 170, 100), ###
        }

        self.color2name = {v: k for k, v in self.name2color.items()}

    def name2id(self, name):
        if name in ['unlabeled', 'water', 'other', 'dynamic']: return 11
        if name in ['building', 'fence', 'bridge']: return 2
        if name in ['pole']: return 6
        if name in ['wall', 'rail_track', 'guard_rail', 'road_line']: return 10
        if name in ['person']: return 5
        if name in ['sky']: return 0
        if name in ['road', 'static', 'sidewalk', 'ground']: return 1
        if name in ['car']: return 2
        if name in ['vegetation']: return 4
        if name in ['traffic_light']: return 7
        if name in ['traffic_sign']: return 8
        if name in ['terrain']: return 3


    def generate_dataset_file(self, args):

        f = open(args.dst_path, 'w+', encoding='utf-8')

        # Test if src_path exists
        if not os.path.exists(args.src_path):
            raise Exception("Source path does not exist.")

        # Iterate through every directory in the src_path
        for dir_name in os.listdir(args.src_path):

            # Open it
            road_rgb_path = os.path.join(args.src_path, dir_name, '1', 'rgb_v')
            if not os.path.exists(road_rgb_path):
                continue
            print(f"Processing {dir_name}...")

            # Iterate through all the pngs in tmp_path
            for file_name in os.listdir(road_rgb_path):
                if file_name.endswith(".png"):
                    # img_path
                    f.write(Path(os.path.join(args.src_path, dir_name, '1', 'rgb_v', file_name)).resolve().__str__())
                    f.write(',')

                    # robust_label_path
                    f.write(Path(os.path.join(args.src_path, dir_name, '1', 'mask_v', file_name)).resolve().__str__())
                    f.write(',')

                    # gbuffer_path
                    # TODO: Make G-Buffer here!
                    g_buffer_dir = Path(args.dst_path, '..', 'g_buffer', dir_name, file_name + '.npz').resolve()

                    if not os.path.exists(g_buffer_dir):

                        print(f"{g_buffer_dir} does not exist, calculating...")
                        #
                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(g_buffer_dir), exist_ok=True)

                        data = {}

                        # Load RGB image
                        data['img'] = np.array(imageio.imread(os.path.join(args.src_path, dir_name, '1', 'rgb_v', file_name)))

                        data['gbuffers'] = np.array([
                            np.array(imageio.imread(os.path.join(args.src_path, dir_name, '1', 'depth_v', file_name)))
                        ])

                        gtlabel_map = np.array(imageio.imread(os.path.join(args.src_path, dir_name, '1', 'mask_v', file_name)))

                        # From (h, w, 3) to (h, w, 1), use lambda function to convert to (h, w, 1)
                        new_gtlabel_map = np.zeros(shape=(gtlabel_map.shape[0], gtlabel_map.shape[1]), dtype=np.uint8)

                        for i in range(gtlabel_map.shape[0]):
                            for j in range(gtlabel_map.shape[1]):
                                new_gtlabel_map[i, j] = np.array([self.name2id(self.color2name[tuple(gtlabel_map[i, j, :])])])

                        shader_map = np.zeros((gtlabel_map.shape[0], gtlabel_map.shape[1], 12), dtype=np.float32)
                        for k in range(12):
                            shader_map[:, :, k] = (new_gtlabel_map == k).astype(np.float32)

                        data['shader'] = shader_map

                        np.savez(g_buffer_dir, **data)

                    f.write(g_buffer_dir.resolve().__str__())
                    f.write(',')

                    # gt_label_path
                    f.write(Path(os.path.join(args.src_path, dir_name, '1', 'mask_v', file_name)).resolve().__str__())
                    f.write('\n')

        f.close()

        # Then, Iterate through all g_buffers and calculate the means and stds

        sum = np.zeros(shape=(720, 1280, 3), dtype=np.float32)
        tmp_std = np.zeros(shape=(720, 1280, 3), dtype=np.float32)

        g_buffer_dir = Path(args.dst_path, '..', 'g_buffer').resolve()

        for filename in glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True):
            data = np.load(filename)
            sum += data['gbuffers'][0]
            print(filename)

        mean = sum / len(list(glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True)))

        for filename in glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True):
            data = np.load(filename)
            tmp_std += (data['gbuffers'][0] - mean) ** 2
            print(filename)

        std = np.sqrt(tmp_std / len(list(glob.iglob(os.path.join(g_buffer_dir, '**/*.npz'), recursive=True))))

        # Save the mean and std
        data = {}
        data['g_m'] = mean
        data['g_s'] = std
        # Mkdir if not exists
        os.makedirs(Path(__file__).parent / 'stats', exist_ok=True)

        np.savez(Path(__file__).parent / 'stats/carla_stats.npz', **data)



if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(description="Generate dataset file for matching.")
    parser.add_argument("--type", type=str, required=True, choices=["cityscape", "carla"], help="Dataset type.")
    parser.add_argument("--src_path", type=str, required=True, help="Path to source dataset.")
    parser.add_argument("--dst_path", type=str, required=True, help="Path to target file.")
    args = parser.parse_args()

    # Generate dataset file
    if args.type == "cityscape":
        dataset = CityScape()

    elif args.type == "carla":
        dataset = Carla()

    else:
        raise Exception("Unknown dataset type.")

    dataset.generate_dataset_file(args)
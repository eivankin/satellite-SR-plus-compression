import os
import json
import glob
import torch
import random
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.utils import data as data
from torch.utils.data import WeightedRandomSampler
import rasterio
import shapely
from rasterio.crs import CRS
from rasterio.warp import transform_geom

from basicsr.utils.registry import DATASET_REGISTRY

from ssr.utils.data_utils import has_black_pixels, get_random_nonzero_extent

random.seed(123)

class CustomWeightedRandomSampler(WeightedRandomSampler):
    """
    WeightedRandomSampler except allows for more than 2^24 samples to be sampled.
    Source code: https://github.com/pytorch/pytorch/issues/2576#issuecomment-831780307
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

def tile_to_point(utm_zone: int, row: int, col: int, size: int = 512, pixel_size: float = 1.25):
    x = (-col * size * pixel_size) - (size * pixel_size / 2)
    y = (row * size * pixel_size) + (size * pixel_size / 2)
    return shapely.Point(y, x)

def utm_to_wgs84(geometry, utm_zone: int):
    try:
        src_crs = CRS.from_epsg(utm_zone)
        dst_crs = CRS.from_epsg(4326)
        return shapely.geometry.shape(transform_geom(src_crs, dst_crs, geometry))
    except Exception as e:
        print(f"Warning: Failed to transform point {geometry} in zone {utm_zone}: {str(e)}")
        return None

@DATASET_REGISTRY.register()
class S2NAIPDataset(data.Dataset):
    """
    Dataset object for the S2NAIP data. Builds a list of Sentinel-2 time series and NAIP image pairs.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            sentinel2_path (str): Data path for Sentinel-2 imagery.
            naip_path (str): Data path for NAIP imagery.
            n_sentinel2_images (int): Number of Sentinel-2 images to use as input to model.
            scale (int): Upsample amount, only 4x is supported currently.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(S2NAIPDataset, self).__init__()
        self.opt = opt

        self.split = opt['phase']
        train = True if self.split == 'train' else False

        # Random cropping and resizing augmentation; training only
        self.rand_crop = opt['rand_crop'] if 'rand_crop' in opt else False

        self.n_s2_images = int(opt['n_s2_images'])
        self.scale = int(opt['scale'])

        # Flags whether the model being used expects [b, n_images, channels, h, w] or [b, n_images*channels, h, w].
        # The L2-based models expect the first shape, while the ESRGAN models expect the latter.
        self.use_3d = opt['use_3d'] if 'use_3d' in opt else False

        # Path to high-res images of older timestamps and corresponding locations to training data.
        # In the case of the S2NAIP dataset, that means NAIP images from 2016-2018.
        self.old_naip_path = opt['old_naip_path'] if 'old_naip_path' in opt else None

        # Path to osm_chips_to_masks.json if provided. 
        self.osm_objs_path = opt['osm_objs_path'] if 'osm_objs_path' in opt else None

        # Sentinel-2 bands to be used as input. Default to just using tci.
        self.s2_bands = opt['s2_bands'] if 's2_bands' in opt else [3, 2, 1] # RGB
        self.naip_bands = opt.get('naip_bands', [0, 1, 2])

        self.plot_inputs = opt.get('plot_inputs', False)

        # If a path to older NAIP imagery is provided, build dictionary of each chip:path to png.
        if self.old_naip_path is not None:
            old_naip_chips = {}
            for old_naip in glob.glob(self.old_naip_path + '/**/*.png', recursive=True):
                old_chip = old_naip.split('/')[-1][:-4]

                if not old_chip in old_naip_chips:
                    old_naip_chips[old_chip] = []
                old_naip_chips[old_chip].append(old_naip)

        # If a path to osm_chips_to_masks.json is provided, we want to filter out datapoints where
        # there is not at least n_osm_objs objects in the NAIP image.
        # if self.osm_chips_to_masks is not None and train:
        #     osm_obj_data = json.load(open(self.osm_chips_to_masks))
        #     print("Loaded osm_chip_to_masks.json with ", len(osm_obj_data), " entries.")

        # Paths to Sentinel-2 and NAIP imagery.
        self.s2_path = opt['sentinel2_path']
        self.naip_path = opt['naip_path']
        if not (os.path.exists(self.s2_path) and os.path.exists(self.naip_path)):
            raise Exception("Please make sure the paths to the data directories are correct.")

        self.naip_chips = glob.glob(self.naip_path + '/**/*.png', recursive=True)
        if self.osm_objs_path is not None:
            self.osm_chips = glob.glob(self.osm_objs_path + '/**/*.geojson', recursive=True)

        # Reduce the training set down to a specified number of samples. If not specified, whole set is used.
        if 'train_samples' in opt and train:
            self.naip_chips = random.sample(self.naip_chips, opt['train_samples'])

        self.datapoints = []
        for n in self.naip_chips:
            # Extract the X,Y chip from this NAIP image filepath.
            split_path = n.split('/')
            chip = split_path[-2]
            tile = split_path[-1][:-4]  # remove the .tif extension

            # If old_hr_path is specified, grab an old high-res image (NAIP) for the current datapoint.
            if self.old_naip_path is not None:
                old_chip = old_naip_chips[chip][0]

            # If using OSM Object ESRGAN, filter dataset to only include images containing OpenStreetMap objects.
            # if self.osm_chips_to_masks is not None and train:
            #     if not (chip in osm_obj_data and sum([len(osm_obj_data[chip][k]) for k in osm_obj_data[chip].keys()]) >= opt['n_osm_objs']):
            #         continue

            # Gather the filepaths to the Sentinel-2 bands specified in the config.
            s2_paths = [os.path.join(self.s2_path, chip, tile + '.tif')]

            osm_path = None
            if self.osm_objs_path is not None:
                osm_path = os.path.join(self.osm_objs_path, chip, tile + '.geojson')

            # Return the low-res, high-res, chip (ex. 12345_67890), and [optionally] older high-res image paths. 
            if self.old_naip_path:
                self.datapoints.append([n, s2_paths, chip, old_chip])
            else:
                self.datapoints.append([n, s2_paths, chip, osm_path])

        self.data_len = len(self.datapoints)
        print("Number of datapoints for split ", self.split, ": ", self.data_len)

    def get_tile_weight_sampler(self, tile_weights):
        weights = []
        for dp in self.datapoints:
            # Extract the NAIP chip from this datapoint's NAIP path.
            # With the chip, we can index into the tile_weights dict (naip_chip : weight)
            # and then weight this datapoint pair in self.datapoints based on that value.
            naip_path = dp[0]
            split = naip_path.split('/')[-1]
            chip = split[:-4]

            # If the chip isn't in the tile weights dict, then there weren't any OSM features
            # in that chip, so we can set the weight to be relatively low (ex. 1).
            if not chip in tile_weights:
                weights.append(1)
            else:
                weights.append(tile_weights[chip])

        print('Using tile_weight_sampler, min={} max={} mean={}'.format(min(weights), max(weights), np.mean(weights)))
        return CustomWeightedRandomSampler(weights, len(self.datapoints))

    def __getitem__(self, index):

        # A while loop and try/excepts to catch a few images that we want to ignore during 
        # training but do not necessarily want to remove from the dataset, such as the
        # ground truth NAIP image being partially invalid (all black).
        osm_path = None
        coordinates = (0.0, 0.0)
        counter = 0
        while True:
            index += counter  # increment the index based on what errors have been caught
            if index >= self.data_len:
                index = 0

            datapoint = self.datapoints[index]

            if self.old_naip_path:
                naip_path, s2_paths, zoom17_tile, old_naip_path = datapoint[0], datapoint[1], datapoint[2], datapoint[3]
            else:
                naip_path, s2_paths, zoom17_tile, osm_path = datapoint[0], datapoint[1], datapoint[2], datapoint[3]

            chip_name = naip_path.split('/')[-1][:-4]

            # --- Extract geo coordinates from chip name using logic from points_from_splits.py ---
            # chip name format: "{utm_zone}_{row}_{col}"
            try:
                chip_parts = chip_name.split("_")
                if len(chip_parts) == 3:
                    utm_zone = int(chip_parts[0])
                    row = int(chip_parts[1])
                    col = int(chip_parts[2])
                    point = tile_to_point(utm_zone, row, col)
                    wgs84_point = utm_to_wgs84(point, utm_zone)
                    if wgs84_point is not None:
                        coordinates = (wgs84_point.y, wgs84_point.x)  # lat, lon
            except Exception as e:
                print(f"Failed to extract coordinates from chip {zoom17_tile}/{chip_name}: {e}")
                coordinates = (0.0, 0.0)

            # --- end geo extraction ---

            # naip_path = "custom_dataset/prepared/train/naip/32614_30_-164/32614_968_-5233.png"

            # Load the 128x128 NAIP chip in as a tensor of shape [channels, height, width].
            # num_naip_bands = min(len(self.s2_bands), 4)
            naip_chip = torchvision.io.read_image(naip_path)[self.naip_bands, :, :]

            rand_hr_x1, rand_hr_x2, rand_hr_y1, rand_hr_y2 = 0, 128, 0, 128
            if self.rand_crop:
                rand_hr_x1, rand_hr_x2, rand_hr_y1, rand_hr_y2 = get_random_nonzero_extent(naip_chip)

            rand_lr_x1, rand_lr_x2, rand_lr_y1, rand_lr_y2 = map(lambda x: int(x / self.scale), [rand_hr_x1, rand_hr_x2, rand_hr_y1, rand_hr_y2])
            naip_chip = naip_chip[:, rand_hr_x1:rand_hr_x2, rand_hr_y1:rand_hr_y2]

            # Check for black pixels (almost certainly invalid) and skip if found.
            if has_black_pixels(naip_chip):
                # counter += 1
                # continue
                # raise ValueError(f"NAIP image {naip_path} contains black pixels")
                ...
            img_HR = naip_chip

            # Load the T*32x32xC S2 files for each band in as a tensor.
            # There are a few rare cases where loading the Sentinel-2 image fails, skip if found.
            try:
                s2_tensor = None
                for i,s2_path in enumerate(s2_paths):

                    # There are tiles where certain bands aren't available, use zero tensors in this case.
                    if not os.path.exists(s2_path):
                        raise ValueError(f"Sentinel-2 file not found: {s2_path}")
                        # img_size = (self.n_s2_images, 3, 32, 32) if 'tci' in s2_path else (self.n_s2_images, 1, 32, 32)
                        # s2_img = torch.zeros(img_size, dtype=torch.uint8)
                    else:
                        # s2_img = torchvision.io.read_image(s2_path)
                        with rasterio.open(s2_path) as src:
                            s2_img = torch.from_numpy(src.read(self.s2_bands))
                            s2_img = s2_img[:, rand_lr_x1:rand_lr_x2, rand_lr_y1:rand_lr_y2]

                        # s2_img = torch.reshape(s2_img, (self.n_s2_images, -1, 32, 32))#.permute(1,0,2,3)
                        # stack groups of 3 channels along n_images axis
                        s2_img = torch.stack([s2_img[(i * 3):(i * 3) + 3] for i in range(self.n_s2_images)])
                        expected_channels = 3
                        expected_shape = (self.n_s2_images, expected_channels, 32, 32)
                        assert s2_img.shape == expected_shape, (s2_img.shape, expected_shape)

                    if i == 0:
                        s2_tensor = s2_img
                    else:
                        s2_tensor = torch.cat((s2_tensor, s2_img), dim=1)
            except Exception as e:
                raise e
                # counter += 1
                # print("s2 failed")
                # continue

            # Skip the cases when there are not as many Sentinel-2 images as requested.
            if s2_tensor.shape[0] < self.n_s2_images:
                counter += 1
                continue

            # Iterate through the 32x32 tci chunks at each timestep, separating them into "good" (valid)
            # and "bad" (partially black, invalid). Will use these to pick best collection of S2 images.
            tci_chunks = s2_tensor[:, :3, :, :]
            goods, bads = [], []
            for i,ts in enumerate(tci_chunks):
                if has_black_pixels(ts):
                    bads.append(i)
                else:
                    goods.append(i)

            # Pick self.n_s2_images random indices of S2 images to use. Skip ones that are partially black.
            if len(goods) >= self.n_s2_images:
                rand_indices = random.sample(goods, self.n_s2_images)
            else:
                need = self.n_s2_images - len(goods)
                rand_indices = goods + random.sample(bads, need)
            rand_indices_tensor = torch.as_tensor(rand_indices)

            # Extract the self.n_s2_images from the first dimension.
            img_S2 = s2_tensor[rand_indices_tensor]

            # If using a model that expects 5 dimensions, we will not reshape to 4 dimensions.
            if not self.use_3d:
                img_S2 = torch.reshape(img_S2, (-1, 32, 32))

            if self.old_naip_path is not None:
                old_naip_chip = torchvision.io.read_image(old_naip_path)
                img_old_HR = old_naip_chip
                return {'hr': img_HR, 'lr': img_S2, 'old_hr': img_old_HR, 'Index': index, 'Phase': self.split, 'Chip': zoom17_tile}

            osm_json = None
            naip_downscale_factor = 2  # S2-NAIP uses coordinates tied to 512x512 for OSM, we want 256x256
            min_size = 5
            if osm_path is not None:
                with open(osm_path) as f:
                    # Convert the format:
                    # 1. extract bbox of each feature,
                    # 2. convert coordinates to int so we can crop images later using them.
                    # Format should be the following: {category_name: [[x1, y1, x2, y2], ...], ...}.
                    # Include only polygon-type (Polygon, MultiPolygon) features like buildings.
                    osm_json_raw = json.load(f)
                    if osm_json_raw:
                        osm_json = {}
                        for feature in osm_json_raw.get('features', []):
                            geom_type = feature.get('geometry', {}).get('type', '')
                            if geom_type not in ['Polygon', 'MultiPolygon']:
                                continue
                            category = feature.get('properties', {}).get('category', 'unknown')
                            # Extract all polygons (MultiPolygon is a list of polygons)
                            coords_list = []
                            if geom_type == 'Polygon':
                                coords_list = [feature['geometry']['coordinates']]
                            elif geom_type == 'MultiPolygon':
                                coords_list = feature['geometry']['coordinates']
                            for poly in coords_list:
                                # poly is a list of linear rings, take the exterior ring (first)
                                exterior = poly[0]
                                xs = [int(round(pt[0] / naip_downscale_factor)) for pt in exterior]
                                ys = [int(round(pt[1] / naip_downscale_factor)) for pt in exterior]
                                if not xs or not ys:
                                    continue
                                x1, x2 = max(min(xs), rand_hr_y1), min(max(xs), rand_hr_y2)
                                if x2 - x1 < min_size:
                                    continue
                                y1, y2 = max(min(ys), rand_hr_x1), min(max(ys), rand_hr_x2)
                                if y2 - y1 < min_size:
                                    continue
                                bbox = [x1 - rand_hr_y1, y1 - rand_hr_x1, x2 - rand_hr_y1, y2 - rand_hr_x1]  # [x1, y1, x2, y2]
                                osm_json.setdefault(category, []).append(bbox)
                            # if max_features > 0 and len(osm_json[category]) >= max_features:
                            #     break
            if self.plot_inputs and osm_json and random.random() < 0.3:
                import matplotlib.pyplot as plt
                # Draw LR, HR and OSM bboxes (different colors for each category)
                # Save plot as png {phase}_{index}.png
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(img_S2.permute(1, 2, 0)[:, :, :3].cpu().numpy())
                axs[0].set_title(f'LR (S2) | crop ({rand_lr_x1}, {rand_lr_y1}, {rand_lr_x2}, {rand_lr_y2}) | channels {self.s2_bands}')
                axs[1].imshow(img_HR.permute(1, 2, 0).cpu().numpy())
                axs[1].set_title(f'HR (NAIP) | crop ({rand_hr_x1}, {rand_hr_y1}, {rand_hr_x2}, {rand_hr_y2}) | channels {self.naip_bands}')

                colors = ['r', 'b', 'y', 'm', 'c']
                for idx, (cat, bboxes) in enumerate(osm_json.items()):
                    color = colors[idx % len(colors)]
                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color,
                                             facecolor='none', label=cat)
                        axs[1].add_patch(rect)
                handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], lw=2, label=cat) for i, cat in
                           enumerate(osm_json.keys())]
                axs[1].legend(handles=handles)

                plt.tight_layout()
                plt.savefig(f'train_examples/{self.split}_{index}.png')
                print(f'Saved {self.split}_{index}.png')
                plt.close(fig)

            return {'hr': img_HR, 'lr': img_S2, 'Index': index, 'Phase': self.split, 'Chip': zoom17_tile, 'osm': json.dumps(osm_json) if osm_json else "{}", "coords": torch.tensor(coordinates), 'chip': chip_name}

    def __len__(self):
        return self.data_len

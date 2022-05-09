import logging
from pathlib import Path

import IPython
import imageio
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import torch

from .batch_types import EPEBatch
from .synthetic import SyntheticDataset
from .utils import mat2tensor, normalize_dim



def center(x, m, s):
	x[0,:,:] = (x[0,:,:] - m[0]) / s[0]
	x[1,:,:] = (x[1,:,:] - m[1]) / s[1]
	x[2,:,:] = (x[2,:,:] - m[2]) / s[2]
	return x


# CityScape ID!
def material_from_gt_label(gt_labelmap):
	""" Merges several classes. """

	h,w = gt_labelmap.shape
	shader_map = np.zeros((h, w, 12), dtype=np.float32)
	for k in range(12):
		shader_map[:, :, k] = (gt_labelmap == k).astype(np.float32)
	return shader_map


class CarlaDataset(SyntheticDataset):
	def __init__(self, paths, transform=None, gbuffers='carla'):
		"""
		paths -- list of tuples with (img_path, robust_label_path, gbuffer_path, gt_label_path)
		"""

		super(CarlaDataset, self).__init__('CarlaDataset')

		assert gbuffers in ['carla']

		self.transform = transform
		self.gbuffers  = gbuffers
		# self.shader    = class_type

		self._paths    = paths
		self._path2id  = {p[0].stem:i for i,p in enumerate(self._paths)}
		if self._log.isEnabledFor(logging.DEBUG):
			self._log.debug(f'Mapping paths to dataset IDs (showing first 30 entries):')
			for i,(k,v) in zip(range(30),self._path2id.items()):
				self._log.debug(f'path2id[{k}] = {v}')
				pass
			pass

		try:
			data = np.load(Path(__file__).parent.parent / 'stats/carla_stats.npz')
			# self._img_mean  = data['i_m']
			# self._img_std   = data['i_s']
			self._gbuf_mean = data['g_m']
			self._gbuf_std  = data['g_s']
			self._log.info(f'Loaded dataset stats.')
		except:
			# self._img_mean  = None
			# self._img_std   = None
			self._gbuf_mean = None
			self._gbuf_std  = None
			pass

		self._log.info(f'Found {len(self._paths)} samples.')
		pass


	@property
	def num_gbuffer_channels(self):
		""" Number of image channels the provided G-buffers contain."""
		return 3


	@property
	def num_classes(self):
		""" Number of classes in the semantic segmentation maps."""
		return 12


	@property
	def cls2gbuf(self):
		if self.gbuffers == 'all':
			# all: just handle sky class differently
			return {\
				0:lambda g:g[:,15:21,:,:]}
		else:
			return {}


	def get_id(self, img_filename):
		return self._path2id.get(Path(img_filename).stem)


	def __getitem__(self, index):

		index  = index % self.__len__()
		img_path, robust_label_path, gbuffer_path, gt_label_path = self._paths[index]

		if not gbuffer_path.exists():
			self._log.error(f'Gbuffers at {gbuffer_path} do not exist.')
			raise FileNotFoundError
			pass

		data = np.load(gbuffer_path)

		# print('gbuffer_path:', gbuffer_path)
		# print('img_path:', img_path)

		img       = mat2tensor(data['img'].astype(np.float32) / 255.0)
		gbuffers  = mat2tensor(data['gbuffers'][0].astype(np.float32))
		gt_labels = mat2tensor(data['shader'].astype(np.float32))

		# print('img', img)
		# print(gbuffers)
		# print(gt_labels)
		# print(img.shape, gbuffers.shape, gt_labels.shape)

		if self._gbuf_mean is not None:
			gbuffers = center(gbuffers, self._gbuf_mean, self._gbuf_std)
			pass

		if not robust_label_path.exists():
			self._log.error(f'Robust labels at {robust_label_path} do not exist.')
			raise FileNotFoundError
			pass

		robust_labels = imageio.imread(robust_label_path)
		robust_labels = torch.zeros((1, 720, 1280), dtype=torch.float32)

		# print(gt_labels.shape)

		for k in range(12):
			robust_labels += gt_labels[k] * k

		# Change dtype to long
		robust_labels = robust_labels.long()

		# print('robust_labels:', robust_labels)

		return EPEBatch(img, gbuffers=gbuffers, gt_labels=gt_labels, robust_labels=robust_labels, path=img_path, coords=None)


	def __len__(self):
		return len(self._paths)

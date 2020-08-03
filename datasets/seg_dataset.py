"""
  
    Created on 2/18/20

    @author: Baoxiong Jia

    Description:

"""
import logging
import pickle

from pathlib import Path
import numpy as np

import torch.utils.data

import datasets.transform as slowfast_transform
import datasets.slowfast_utils as slowfast_datautils
from datasets.build import DATASET_REGISTRY

import datasets.utils as data_utils
from datasets.metadata import Metadata

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Lemma_seg(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, subclass='all'):
        assert mode in ['train', 'val', 'test', 'all'], 'Split [{}] not supported for Lemma_seg'.format(mode)
        self._mode = mode
        self._cfg = cfg
        self._video_meta = {}
        self._task = cfg.EXP.TASK
        self._view = cfg.EXP.VIEW_TYPE
        self._img_type = cfg.EXP.IMG_TYPE
        self._label_type = cfg.EXP.LABEL_TYPE
        self._video_type = cfg.EXP.VIDEO_TYPE
        self._num_classes = cfg.MODEL.NUM_CLASSES

        # Segment sampling parameters
        self._sample_rate = cfg.DATA.SAMPLING_RATE
        self._clip_length = cfg.DATA.NUM_FRAMES
        self._window_len = self._clip_length * self._sample_rate

        # Normalization parameters
        self._data_mean = cfg.DATA.DATA_MEAN
        self._data_std = cfg.DATA.DATA_STD
        if self._img_type == 'rgb':
            self._use_bgr = cfg.DATA.BGR
        else:
            self._use_bgr = False

        self._crop_size = cfg.DATA.CROP_SIZE
        if self._mode == 'train':
            self._jitter_min_scale = cfg.DATA.TRAIN_JITTER_SCALES[0]
            self._jitter_max_scale = cfg.DATA.TRAIN_JITTER_SCALES[1]
            self._use_color_augmentation = cfg.TRAIN.USE_COLOR_AUGMENTATION
            self._pca_jitter_only = cfg.TRAIN.PCA_JITTER_ONLY
            self._pca_eigval = cfg.TRAIN.PCA_EIGVAL
            self._pca_eigvec = cfg.TRAIN.PCA_EIGVEC
        else:
            self._test_force_flip = cfg.TEST.FORCE_FLIP

        logger.info('Constructing Lemma_seg {}'.format(mode))
        self._load_data(verbose=cfg.DEBUG)


    def _load_data(self, verbose=False):
        self._frame_paths = {}
        self._gt_bboxes_labels = {}
        self._bboxes_labels = {}
        self._gt_tasks = {}
        self._vid_idx_to_name = None
        self._selected_indices = list()

        # Load Image, GT, Detections
        keys = ['frame_paths', 'gt_bboxes_labels', 'gt_tasks', 'bboxes_labels', 'vid_idx_to_name', 'vid_name_to_idx']
        if not verbose:
            anno_rel_path = 'full'
            split_file = Path(self._cfg.DATA.GT_LIST_DIR) / 'splits' / '{}_{}.json'.format(self._cfg.EXP.VIDEO_TYPE,
                                                                                           self._mode)
            cache = self._cfg.TRAIN.CACHE
            cache_path = Path(self._cfg.DATA.SAVE_PATH) / '{}_{}_{}_{}'.format(self._task, self._view,
                                                                               self._img_type, self._video_type)
        else:
            anno_rel_path = 'unit_test'
            split_file = None
            cache = False
            cache_path = None

        if cache and cache_path.exists():
            for key in keys:
                with (cache_path / '{}.p'.format(key)).open('rb') as f:
                    val = pickle.load(f)
                    setattr(self, '_{}'.format(key), val)
        else:
            gt_path = Path(self._cfg.DATA.GT_LIST_DIR) / anno_rel_path / '{}_{}_frames.p'.format(self._task, self._view)
            (
                self._frame_paths,
                self._gt_bboxes_labels,
                self._vid_idx_to_name,
                self._vid_name_to_idx
            ) = data_utils.parse_gt_result(gt_path, self._cfg)

            if cache and not cache_path.exists():
                cache_path.mkdir(exist_ok=True, parents=True)
                for key in keys:
                    with (cache_path / '{}.p'.format(key)).open('wb') as f:
                        pickle.dump(getattr(self, '_{}'.format(key)), f)

        # Load embeddings
        with Path(self._cfg.EMBED_PATH).open('rb') as f:
            embeddings = pickle.load(f)
        self._action_embeddings = embeddings['action']
        self._task_embeddings = embeddings['task']

        self._selected_indices = data_utils.get_selected_indices(self._frame_paths, self._gt_bboxes_labels,
                                                                 self._vid_name_to_idx, self._cfg, self._mode,
                                                                 anno_rel_path, split_file=split_file)
        self.print_summary()


    def print_summary(self):
        logger.info('=== MICASA Segment dataset summary ===')
        logger.info('Split: {}, View: {}, Src: {}, Task: {}'.format(self._mode, self._view,
                                                                    self._img_type, self._label_type))
        logger.info('Number of total videos: {}'.format(len(self._frame_paths)))
        total_frames = sum(
            len(frame_path) for _, frame_path in self._frame_paths.items()
        )
        logger.info('Number of total frames: {}'.format(total_frames))
        logger.info('Number of selected frames: {}'.format(len(self)))

        if self._view == 'tpv':
            total_boxes = sum(
                len(self._gt_bboxes_labels[self._vid_idx_to_name[vid_idx]][center_idx])
                for vid_idx, center_idx, _ in self._selected_indices
            )
            logger.info('Number of used boxes: {}'.format(total_boxes))


    def __len__(self):
        return len(self._selected_indices)


    def __getitem__(self, index, verbose=False):
        if self._mode != 'train':
            vid_idx, center_idx, _ = self._selected_indices[index]
        else:
            vid_idx, center_idx, segment_length = self._selected_indices[index]
            start = data_utils.temporal_sample_offset(segment_length, self._window_len)
            center_idx = center_idx + start

        vid_name = self._vid_idx_to_name[vid_idx]

        seq = slowfast_datautils.get_sequence(
            center_idx,
            self._window_len // 2,
            self._sample_rate,
            num_frames=len(self._frame_paths[vid_name]),
        )

        image_paths = [[str(Path(self._cfg.DATA.PATH_TO_DATA_DIR) / x)
                        for x in self._frame_paths[vid_name][frame]] for frame in seq]
        imgs = data_utils.get_frames(image_paths, src=self._img_type)

        assert(vid_name in self._gt_bboxes_labels.keys() and center_idx in self._gt_bboxes_labels[vid_name].keys()), \
            '{} has no frame {}'.format(vid_name, center_idx)
        bboxes_labels = self._gt_bboxes_labels[vid_name][center_idx]
        pos_max = 3
        act_max = 3

        label_dict = dict()
        label_dict['verb'] = torch.zeros([len(bboxes_labels), len(Metadata.action)])
        label_dict['noun'] = torch.zeros([len(bboxes_labels), len(Metadata.object)])
        label_dict['hoi'] = torch.zeros([len(bboxes_labels), len(Metadata.hoi)])
        action_labels = (torch.ones([len(bboxes_labels), act_max, 1]).type(torch.LongTensor) * -1.).type(torch.long)
        object_labels = torch.zeros([len(bboxes_labels), act_max, pos_max, len(Metadata.object)])
        task_labels = torch.zeros([len(bboxes_labels), act_max, len(Metadata.task)])
        num_actions = torch.zeros([len(bboxes_labels), 1])
        num_objs = torch.zeros([len(bboxes_labels), act_max, 1])
        bboxes = torch.zeros([len(bboxes_labels), 4])

        if self._mode == 'train':
            action_embeddings = torch.zeros([len(bboxes_labels), act_max, self._action_embeddings[0].shape[0]])
            task_embeddings = torch.zeros([len(bboxes_labels), act_max, self._task_embeddings[0].shape[0]])
        else:
            action_embeddings = torch.cat([
                torch.cat([
                        torch.tensor(self._action_embeddings).unsqueeze(0)
                    for _ in range(act_max)], dim=0).unsqueeze(0)
                for _ in range(len(bboxes_labels))], dim=0).type(torch.float32)
            task_embeddings = torch.cat([
                torch.cat([
                            torch.tensor(self._task_embeddings).unsqueeze(0)
                    for _ in range(act_max)], dim=0).unsqueeze(0)
                for _ in range(len(bboxes_labels))], dim=0).type(torch.float32)

        for bbox_idx, (bbox, labels, tasks, hois, pid) in enumerate(bboxes_labels):
            num_actions[bbox_idx] = len(labels)
            labels = sorted(labels, key=lambda x: x[0])
            for a_idx, label in enumerate(labels):
                action = label[0]
                action_labels[bbox_idx][a_idx] = int(action)
                label_dict['verb'][bbox_idx][action] = 1.0
                objs = label[1]
                num_objs[bbox_idx][a_idx] = len(objs)
                for pos_idx, obj_idx in enumerate(objs):
                    object_labels[bbox_idx][a_idx][pos_idx].scatter_(0, torch.tensor(obj_idx), 1.0)
                    label_dict['noun'][bbox_idx][obj_idx] = 1.0
                if self._mode == 'train':
                    action_embeddings[bbox_idx][a_idx] = torch.tensor(self._action_embeddings[action])
                for task_idx in tasks[a_idx]:
                    task_labels[bbox_idx][a_idx][task_idx] = 1.0
                if self._mode == 'train':
                    task_embeddings[bbox_idx][a_idx] = torch.mean(
                            torch.tensor([self._task_embeddings[tsk] for tsk in tasks[a_idx]]),
                        dim=0, keepdim=True)
                label_dict['hoi'][bbox_idx][hois[a_idx]] = 1.0
            bboxes[bbox_idx] = torch.tensor(bbox).type(torch.float32)
        labels = label_dict[self._cfg.EXP.LABEL_TYPE]
        bboxes = bboxes.numpy()
        ori_bboxes = bboxes.copy()
        metadata = [[vid_idx, center_idx, x[4]] for x in bboxes_labels]

        if verbose:
            data_utils.visualize_sequence(imgs[seq.index(center_idx)], bboxes, labels, self._cfg)

        # Preprocess images and boxes
        # From T H W C -> T C H W
        imgs = imgs.permute(0, 3, 1, 2)
        imgs, bboxes = self._images_and_boxes_preprocessing(imgs, boxes=bboxes)
        # From T C H W -> C T H W
        imgs = imgs.permute(1, 0, 2, 3)
        imgs = slowfast_datautils.pack_pathway_output(self._cfg, imgs)

        meta = dict()
        meta['metadata'] = metadata

        if self._view == 'tpv':
            meta['boxes'] = bboxes
            meta['ori_boxes'] = ori_bboxes

        meta['num_actions'] = num_actions
        meta['action_labels'] = action_labels
        meta['obj_labels'] = object_labels
        meta['num_objects'] = num_objs

        if self._cfg.EXP.SUPERVISION != 'none' or self._cfg.EXP.MODEL_TYPE == 'composed':
            meta['num_actions'] = num_actions
            if self._cfg.EXP.SUPERVISION != 'none':
                meta['task_labels'] = task_labels

        if self._cfg.EXP.MODEL_TYPE != 'plain':
            meta['task_embed'] = task_embeddings
            meta['action_embed'] = action_embeddings

        return imgs, labels, index, meta

    def _images_and_boxes_preprocessing(self, imgs, boxes):
        """
        This function performs preprocessing for the input images and
        corresponding boxes for one clip.

        Args:
            imgs (tensor): the images.
            boxes (ndarray): the boxes for the current clip.

        Returns:
            imgs (tensor): list of preprocessed images.
            boxes (ndarray): preprocessed boxes.
        """
        # Image [0, 255] -> [0, 1].
        imgs = imgs.float()
        imgs = imgs / 255.0

        height, width = imgs.shape[2], imgs.shape[3]
        # The format of boxes is [x1, y1, x2, y2]. The input boxes are in the
        # range of [0, 1].
        if boxes is not None:
            boxes = slowfast_transform.clip_boxes_to_image(boxes, height, width)

        if self._mode == "train":
            # Train split
            imgs, boxes = slowfast_transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = slowfast_transform.random_crop(
                imgs, self._crop_size, boxes=boxes
            )

            # Random flip.
            imgs, boxes = slowfast_transform.horizontal_flip(0.5, imgs, boxes=boxes)
        elif self._mode == "val":
            # Val split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = slowfast_transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = slowfast_transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = slowfast_transform.horizontal_flip(1, imgs, boxes=boxes)
        elif self._mode == "test" or self._mode == 'all':
            # Test split
            # Resize short side to crop_size. Non-local and STRG uses 256.
            imgs, boxes = slowfast_transform.random_short_side_scale_jitter(
                imgs,
                min_size=self._crop_size,
                max_size=self._crop_size,
                boxes=boxes,
            )

            # Apply center crop for val split
            imgs, boxes = slowfast_transform.uniform_crop(
                imgs, size=self._crop_size, spatial_idx=1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = slowfast_transform.horizontal_flip(1, imgs, boxes=boxes)
        else:
            raise NotImplementedError(
                "{} split not supported yet!".format(self._split)
            )

        # Do color augmentation (after divided by 255.0).
        if self._mode == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = slowfast_transform.color_jitter(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = slowfast_transform.lighting_jitter(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        # Normalize images by mean and std.
        imgs = slowfast_transform.color_normalization(
            imgs,
            np.array(self._data_mean, dtype=np.float32),
            np.array(self._data_std, dtype=np.float32),
        )

        if not self._use_bgr:
            # Convert image format from BGR to RGB.
            # Note that Kinetics pre-training uses RGB!
            imgs = imgs[:, [2, 1, 0], ...]

        if boxes is not None:
            boxes = slowfast_transform.clip_boxes_to_image(
                boxes, self._crop_size, self._crop_size
            )

        return imgs, boxes

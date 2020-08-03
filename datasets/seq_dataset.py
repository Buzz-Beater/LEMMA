"""
  
    Created on 2/26/20

    @author: Baoxiong Jia

    Description:

"""

import logging
import pickle
import json
from pathlib import Path
import numpy as np

import torch.utils.data

from datasets.build import DATASET_REGISTRY

import datasets.utils as data_utils
from datasets.metadata import Metadata

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Lemma_seq(torch.utils.data.Dataset):

    def __init__(self, cfg, mode, subclass='all'):
        assert mode in ['train', 'val', 'test', 'all'], 'Split [{}] not supported for Lemma_seg'.format(mode)
        self._mode = mode
        self._cfg = cfg
        self._subclass = subclass
        self._task = 'pred'
        self.feat_path = Path(cfg.feat_path) / 'features'
        self._label_type = cfg.EXP.LABEL_TYPE
        self._num_classes = cfg.MODEL.NUM_CLASSES

        logger.info('Constructing Lemma_seq {} {}'.format(mode, subclass))
        self._load_data(verbose=cfg.DEBUG)

    def _load_data(self, verbose=False):
        if not verbose:
            anno_rel_path = 'full'
            split_file = Path(self._cfg.DATA.GT_LIST_DIR) / 'splits' / '{}_{}.json'.format(self._subclass, self._mode)
            cache = self._cfg.TRAIN.CACHE
            cache_path = Path(self._cfg.DATA.SAVE_PATH) / '{}_tpv'.format(self._task)
        else:
            anno_rel_path = 'unit_test'
            split_file = None
            cache = False
            cache_path = None

        keys = ['frame_paths', 'gt_bboxes_labels', 'gt_tasks', 'bboxes_labels', 'vid_idx_to_name', 'vid_name_to_idx']
        gt_path = Path(self._cfg.DATA.GT_LIST_DIR) / anno_rel_path / '{}_tpv_frames.p'.format(self._task)
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

        if split_file is None:
            self._vid_names = ['28l-12-1-2']
        else:
            with split_file.open('r') as f:
                vids = json.load(f)
            self._vid_names = vids

        with Path('/home/baoxiong/vid_names.json').open('w') as f:
            json.dump(self._vid_names, f)

        print('Total: {} of videos used for {}'.format(len(self), self._mode))

    def __len__(self):
        return len(self._vid_names)


    def __getitem__(self, index, verbose=False):
        vid_prefix = self._vid_names[index]
        fpv_feature_path = self.feat_path / vid_prefix
        pids = sorted([pid for pid in fpv_feature_path.iterdir()])

        all_labels = []
        all_task_labels = []
        all_fpv_features = []
        all_tpv_features = []

        for p_idx, pid in enumerate(pids):
            fpv_features = torch.tensor(np.load(str(pid / 'fpv' / 'features.npy'))).unsqueeze(1)
            frame_ids = np.load(str(pid / 'fpv' / 'frame_ids.npy'))
            labels = []
            task_labels = []
            vid_name = vid_prefix + '|{}'.format(pid.stem)

            for frame_idx, frame in enumerate(frame_ids):
                gt_bbox_labels = self._gt_bboxes_labels[vid_name][frame]
                for gt_bbox_label in gt_bbox_labels:
                    labels.append(torch.zeros(1, len(Metadata.hoi)).scatter_(
                        -1, torch.LongTensor(gt_bbox_label[-2]).unsqueeze(0), 1))
                    task_labels.append(torch.zeros(1, len(Metadata.task)).scatter_(
                        -1, torch.LongTensor([y for x in gt_bbox_label[-3] for y in x ]).unsqueeze(0), 1))

            hoi_labels = torch.cat(labels, dim=0).unsqueeze(1)
            task_labels = torch.cat(task_labels, dim=0).unsqueeze(1)
            all_labels.append(hoi_labels)
            all_fpv_features.append(torch.sum(fpv_features, dim=2))
            all_task_labels.append(task_labels)

            if self._cfg.use_extra and self._cfg.extra == 'tpv':
                tpv_features = torch.tensor(np.load(str(pid / 'tpv' / 'features.npy'))).unsqueeze(1)
                all_tpv_features.append(tpv_features)

        meta = dict()
        all_fpv_features = torch.cat(all_fpv_features, dim=1)
        all_labels = torch.cat(all_labels, dim=1)
        all_task_labels = torch.cat(all_task_labels, dim=1)

        if self._cfg.use_extra and self._cfg.extra == 'tpv':
            lens = [x.size(0) for x in all_tpv_features]

            # TPV/FPV mismatch by 1 frame
            if len(set(lens)) > 1:
                minimum = min(lens)
                for idx, _ in enumerate(all_tpv_features):
                    all_tpv_features[idx] = all_tpv_features[idx][:minimum]

            meta['extra_features'] = torch.cat(all_tpv_features, dim=1)
        else:
            meta['extra_features'] = all_fpv_features

        if meta['extra_features'].size(0) != all_fpv_features.size(0):
            minimum = min(meta['extra_features'].size(0), all_fpv_features.size(0))
            meta['extra_features'] = meta['extra_features'][:minimum]
            all_fpv_features = all_fpv_features[:minimum]

        meta['task_labels'] = all_task_labels
        return all_fpv_features, all_labels, index, meta
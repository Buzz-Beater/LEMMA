"""

    Created on Jan 14, 2020

    @author: Baoxiong Jia

    Annotation parser for action-verb annotations

"""
import json
from pathlib import Path

from config.config import Config

class Metadata(object):
    action = json.load((Path(Config().metadata_path) / 'all_acts.json').open('r'))
    action_index = {v : k for k, v in enumerate(action)}

    object = json.load((Path(Config().metadata_path / 'all_objs.json')).open('r'))
    object_index = {v : k for k, v in enumerate(object)}

    task = json.load((Path(Config().metadata_path) / 'all_tasks.json').open('r'))
    task_index = {v : k for k, v in enumerate(task)}

    hoi = json.load((Path(Config().metadata_path) / 'all_hois.json').open('r'))
    hoi_index = {v : k for k, v in enumerate(hoi)}
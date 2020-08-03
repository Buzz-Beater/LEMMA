"""
  
    Created on 2/15/20

    @author: Baoxiong Jia

    Description:

"""

from pathlib import Path

class Config(object):
    def __init__(self):
        self.project_path = Path('/home/baoxiong/Projects/DOME/experiments/')
        self.annotation_path = Path('/home/baoxiong/Datasets/annotations')
        self.metadata_path = Path('/home/baoxiong/Datasets/metadata')
        self.tmp_path = self.project_path / 'tmp'
        self.intermediate_path = self.project_path / 'intermediate'

        if not self.tmp_path.exists():
            self.tmp_path.mkdir(exist_ok=True, parents=True)
        if not self.intermediate_path.exists():
            self.intermediate_path.mkdir(exist_ok=True, parents=True)

        self.data_path = Path('/home/baoxiong/HDD/Datasets/DOME/Flow')


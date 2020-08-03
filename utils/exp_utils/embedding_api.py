"""
  
    Created on 2/21/20

    @author: Baoxiong Jia

    Description:

"""
import pickle
import re

import numpy as np

from config.config import Config
from datasets.metadata import Metadata

def main(cfg):
    save_path = cfg.intermediate_path / 'embeddings'
    glove_embedding = {}
    with (save_path / 'glove.6B.300d.txt').open('r') as f:
        contents = f.readlines()
    for line in contents:
        splits = line.split()
        word = splits[0]
        embedding = np.array([float(val) for val in splits[1:]])
        glove_embedding[word] = embedding

    action_embeddings = []
    task_embeddings = []
    pattern = '\-| |\)|\('
    for action in Metadata.action:
        words = re.split(pattern, action)
        all = np.array([glove_embedding[word] for word in words])
        action_embedding = np.mean(all, axis=0)
        action_embeddings.append(action_embedding)

    for task in Metadata.task:
        words = re.split(pattern, task)
        all = np.array([glove_embedding[word] for word in words])
        task_embedding = np.mean(all, axis=0)
        task_embeddings.append(task_embedding)

    embedding_meta = {
        'action': action_embeddings,
        'task': task_embeddings
    }

    with (save_path / 'embedding.p').open('wb') as f:
        pickle.dump(embedding_meta, f)



if __name__ == '__main__':
    cfg = Config()
    main(cfg)
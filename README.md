# LEMMA

This repo contains code for our ECCV 2020 paper.

[LEMMA: A Multi-view Dataset for <ins>LE</ins>arning <ins>M</ins>ulti-agent <ins>M</ins>ulti-task <ins>A</ins>ctivities](https://arxiv.org/pdf/2007.15781.pdf)

Baoxiong Jia, Yixin Chen, Siyuan Huang, Yixin Zhu, Song-Chun Zhu

*Proceedings of the European Conference on Computer Vision (ECCV)*, 2020

# Dataset

For the dataset, please check out our [website](https://sites.google.com/view/lemma-activity) for details.

![overview](https://buzz-beater.github.io/assets/publications/2020_lemma_eccv/overview.jpg)

# Dependencies

Our code is adapted from the [SlowFast network repo](https://github.com/facebookresearch/SlowFast), please check that the basic requirements from this repo are satisfied.

Additional dependencies include:
* numpy
* torch
* pathlib
* Pillow
* ffmpeg-python
* opencv-python


# Experiments

This repo contains code for the baseline experiments conducted for **compositional action recognition** and **action/task anticipation**. 

We provide basic configuration files for different architectures and views. Default configurations can be found in ```config/defaulty.py ```. On top of it, we provide sample configurations customized for differnt view and model in ```settings/{model_type}_{view_type}_plain_hoi.yaml```. Other parameters can be found in the two main entrance, ```rec_main.py``` and ```pred_main.py``` respectively for recognition and anticipation task.

For the recognition task, we used word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/). The extracted embeddings for verbs/nouns used in this project are provided in ```embedding.p```. Please set the cfg.EMBED_PATH correctly to use the embeddings.

A sample command for running the experiments would be
```
python rec_main.py --cfg settings/slowfast_tpv_plain_hoi.yaml EXP.MODEL_TYPE branch EXP.VIEW_TYPE tpv
```
This is running the Slowfast-branching model on third-person-view recordings reported in the paper. You might as well change the running settings according to your environment, such arguments in include ```NUM_GPUS```, ```TRAIN.BATCH_SIZE```, ```TEST.BATCH_SIZE```, etc.. Paths for the dataset should be adjusted according to your local arrangement, this include data path, annotations and intermediate paths. You can search for ```/home/baoxiong/...``` for all the paths we used and switch them to the corresponding ones on your local machines.

For the anticipation task, we use extracted features from plain slowfast model. 

A smple command for running the anticipation would be
```
python pred_main.py --model featurebank --use_extra True --extra tpv --task False 
```
This is running the LFG model with extra tpv features provided for the action anticipation task. Similarly, paths should be correctly set to run this experiment.

# Citation

If you find the paper and/or the code helpful, please cite
```
@inproceedings{jia2020lemma,
    author={Jia, Baoxiong and Chen, Yixin and Huang, Siyuan and Zhu, Yixin and Zhu, Song-Chun}, 
    title={LEMMA: A Multiview Dataset for Learning Multi-agent Multi-view Activities}, 
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)}, 
    year={2020}
}
```

# Acknowledgement

We  thank  (i)  Tao  Yuan  at  UCLA  for  designing  the  an-notation  tool,  (ii)  Lifeng  Fan,  Qing  Li,  Tengyu  Liu  at  UCLA  and  ZhouqianJiang for helpful discussions, and (iii) colleagues from UCLA VCLA for assistingthe endeavor of post-processing this massive dataset.

We also gratefully thank the authors from [SlowFast network](https://github.com/facebookresearch/SlowFast) for open-sourcing their code.



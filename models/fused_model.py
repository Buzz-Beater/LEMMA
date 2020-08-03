"""
  
    Created on 2/22/20

    @author: Baoxiong Jia

    Description:

"""
import torch
import torch.nn as nn

import models.video_model_builder as backbone_models
from models.build import MODEL_REGISTRY
from datasets.metadata import Metadata

@MODEL_REGISTRY.register()
class FusedModel(nn.Module):
    def __init__(self, cfg):
        super(FusedModel, self).__init__()
        self.backbone = getattr(backbone_models, cfg.MODEL.BACKBONE_NAME)(cfg)
        self.customized = (cfg.MODEL.MODEL_NAME == 'FusedModel' and cfg.EXP.LABEL_TYPE == 'action')
        self.detection = (cfg.EXP.VIEW_TYPE == 'tpv')
        self.readout = FusedHead(cfg)

    def forward(self, input, meta=None):
        if self.detection:
            if not isinstance(meta, dict):
                x = self.backbone(input, bboxes=meta)
            else:
                x = self.backbone(input, bboxes=meta['boxes'])
        else:
            x = self.backbone(input)
        label_logits, act_logits, obj_logits, task_logits, label_features = self.readout(x, meta)
        return label_logits, act_logits, obj_logits, task_logits, label_features

class FusedHead(nn.Module):
    def __init__(self, cfg):
        super(FusedHead, self).__init__()
        self.pretrain = cfg.MODEL.PRETRAIN_FUSED
        self.model_type = cfg.EXP.MODEL_TYPE
        self.label_type = cfg.EXP.LABEL_TYPE

        if cfg.MODEL.BACKBONE_NAME == 'SlowFast':
            dim_in = cfg.RESNET.WIDTH_PER_GROUP * 32 + cfg.RESNET.WIDTH_PER_GROUP * 32 // cfg.SLOWFAST.BETA_INV
        else:
            dim_in = cfg.RESNET.WIDTH_PER_GROUP * 32

        self.num_classes = cfg.MODEL.NUM_CLASSES
        self.supervision = cfg.EXP.SUPERVISION


        self.object_dim = dim_in
        self.embed_dim = 100

        if cfg.EXP.SUPERVISION != 'none':
            self.object_dim += self.embed_dim
        if cfg.EXP.MODEL_TYPE != 'plain':
            if cfg.EXP.MODEL_TYPE == 'fused':
                self.object_dim += self.embed_dim
            label_dim = self.object_dim + dim_in
        else:
            label_dim = dim_in

        self.label_projection = nn.Linear(label_dim, self.num_classes)

        self.object_num = 3
        self.action_num = 3

        for act_idx in range(self.action_num):
            setattr(self, 'act_feat_{}'.format(act_idx), nn.Linear(dim_in, dim_in))
            setattr(self, 'obj_feat_{}'.format(act_idx), nn.Linear(dim_in, dim_in))
            setattr(self, 'tsk_feat_{}'.format(act_idx), nn.Linear(dim_in, dim_in))

        if cfg.EXP.SUPERVISION != 'none':
            self.task_projection = nn.Linear(dim_in, len(Metadata.task))

        if cfg.EXP.MODEL_TYPE != 'plain':
            self.act_projection = nn.Linear(dim_in, len(Metadata.action))
            self.act_sigmoid = nn.Sigmoid()
            self.task_sigmoid = nn.Sigmoid()
            self.action_embed = nn.Linear(300, self.embed_dim)
            self.task_embed = nn.Linear(300, self.embed_dim)
            self.object_projection = nn.Linear(dim_in, len(Metadata.object))

            for obj_idx in range(self.object_num):
                setattr(self, 'object_pos_{}'.format(obj_idx), nn.Linear(self.object_dim, dim_in))


    def forward(self, x, meta=None):
        act_logits = []
        task_logits = []
        object_logits = []

        visual_features = x
        act_features = torch.cat([
            getattr(self, 'act_feat_{}'.format(act_idx))(x.unsqueeze(1)) for act_idx in range(self.action_num)
        ], dim=-2)
        obj_features = torch.cat([
            getattr(self, 'obj_feat_{}'.format(act_idx))(x.unsqueeze(1)) for act_idx in range(self.action_num)
        ], dim=-2)

        if self.supervision != 'none':
            task_features = torch.cat([
                getattr(self, 'tsk_feat_{}'.format(act_idx))(x.unsqueeze(1)) for act_idx in range(self.action_num)
            ], dim=-2)
            task_embeddings = self.task_embed(meta['task_embed'])
            task_logits = self.task_projection(task_features)

            if self.model_type != 'plain':
                if self.training:
                    obj_features = torch.cat((task_embeddings, obj_features), dim=-1)
                else:
                    task_scores = self.task_sigmoid(task_logits)
                    _, selected_tasks = torch.topk(task_scores, 2, dim=-1)
                    selected_tasks = selected_tasks.unsqueeze(-1).repeat(1, 1, 1, task_embeddings.size(-1))
                    beam_task_embeddings = torch.mean(torch.gather(task_embeddings, 2, selected_tasks), dim=2)
                    obj_features = torch.cat((beam_task_embeddings, obj_features), dim=-1).unsqueeze(2)

        if self.model_type == 'plain':
            label_logits = self.label_projection(visual_features)
            label_features = visual_features
        else:
            act_logits = self.act_projection(act_features)
            if self.model_type == 'fused':
                act_embeddings = self.action_embed(meta['action_embed'])

                if self.training:
                    beam_action_embeddings = act_embeddings
                else:
                    selected_actions = torch.argmax(act_logits, dim=-1, keepdim=True)
                    selected_actions = selected_actions.unsqueeze(-1).repeat(1, 1, 1, act_embeddings.size(-1))
                    beam_action_embeddings = torch.gather(act_embeddings, 2, selected_actions).squeeze(2)
                obj_features = torch.cat((obj_features, beam_action_embeddings), dim=-1).unsqueeze(2)
            else:
                obj_features = obj_features.unsqueeze(2)

            object_features = torch.cat([getattr(self, 'object_pos_{}'.format(obj_idx))(obj_features)
                                             for obj_idx in range(self.object_num)], dim=2)
            object_logits = self.object_projection(object_features)
            # Fusing all object branch and action branch
            label_features = torch.cat((act_features, torch.sum(obj_features, dim=2)), dim=-1)
            label_logits = self.label_projection(torch.sum(label_features, dim=1))

        return label_logits, act_logits, object_logits, task_logits, label_features


        # if self.model_type == 'plain':
        #     label_logits = self.label_projection(visual_features)
        #     if self.supervision != 'none':
        #         task_logits = self.task_projection(act_features)
        #     label_features = torch.sum(act_features, 1)
        # else:
        #     # Multi-tasking model
        #     if self.model_type == 'separate':
        #         act_logits = self.act_projection(act_features)
        #         obj_logits = torch.cat([getattr(self, 'object_pos_{}'.format(obj_idx))(obj_features)
        #                                        for obj_idx in range(self.object_num)], dim=2)
        #
        #     if self.label_type != 'noun':
        #         act_logits = self.act_projection(act_features)
        #         if self.supervision != 'none':
        #             task_logits = self.task_projection(act_features)
        #     if meta is not None and self.label_type != 'verb':
        #         act_embeddings = self.action_embed(meta['action_embed'])
        #         if self.supervision != 'none':
        #             task_logits = self.task_projection(act_features)
        #             task_embeddings = self.task_embed(meta['task_embed'])
        #             if self.training:
        #                 if self.label_type == 'noun':
        #                     object_features = torch.cat((task_embeddings, act_features),
        #                                                 dim=-1).unsqueeze(2).repeat(1, 1, self.object_num, 1)
        #                 else:
        #                     object_features = torch.cat((act_embeddings, task_embeddings, act_features),
        #                                                 dim=-1).unsqueeze(2).repeat(1, 1, self.object_num, 1)
        #             else:
        #                 task_scores = self.task_sigmoid(task_logits)
        #                 _, selected_tasks = torch.topk(task_scores, 2, dim=-1)
        #                 selected_tasks = selected_tasks.unsqueeze(-1).repeat(1, 1, 1, task_embeddings.size(-1))
        #                 beam_task_embeddings = torch.mean(torch.gather(task_embeddings, 2, selected_tasks), dim=2)
        #                 if self.label_type != 'noun':
        #                     act_scores = self.act_sigmoid(act_logits)
        #                     selected_actions = torch.argmax(act_scores, dim=-1, keepdim=True)
        #                     selected_actions = selected_actions.unsqueeze(-1).repeat(1, 1, 1, act_embeddings.size(-1))
        #                     beam_action_embeddings = torch.gather(act_embeddings, 2, selected_actions).squeeze(2)
        #                     object_features = torch.cat((beam_action_embeddings, beam_task_embeddings, act_features),
        #                                                                                             dim=-1).unsqueeze(2)
        #                 else:
        #                     object_features = torch.cat((beam_task_embeddings, act_features), dim=-1).unsqueeze(2)
        #         else:
        #             if self.training:
        #                 if self.label_type == 'noun':
        #                     object_features = act_features.unsqueeze(2)
        #                 else:
        #                     object_features = torch.cat([act_embeddings, act_features], dim=-1).unsqueeze(2)
        #             else:
        #                 act_scores = self.act_sigmoid(act_logits)
        #                 if self.label_type != 'noun':
        #                     selected_actions = torch.argmax(act_scores, dim=-1, keepdim=True)
        #                     selected_actions = selected_actions.unsqueeze(-1).repeat(1, 1, 1, act_embeddings.size(-1))
        #                     beam_action_embeddings = torch.gather(act_embeddings, 2, selected_actions).squeeze(2)
        #                     object_features = torch.cat((beam_action_embeddings, act_features), dim=-1).unsqueeze(2)
        #                 else:
        #                     object_features = act_features.unsqueeze(2)
        #
        #         object_features = torch.cat([getattr(self, 'object_pos_{}'.format(obj_idx))(object_features)
        #                                        for obj_idx in range(self.object_num)], dim=2)
        #         object_logits = self.object_projection(object_features)
        #
        #         label_features = object_features
        #         label_logits = self.label_projection(torch.sum(torch.sum(label_features, dim=2), dim=1))
        #
        # return label_logits, act_logits, task_logits, object_logits, label_features
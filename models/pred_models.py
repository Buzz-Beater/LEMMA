"""
  
    Created on 2/27/20

    @author: Baoxiong Jia

    Description:

"""
import torch
import torch.nn as nn
from datasets.metadata import Metadata

class FusedLSTM(nn.Module):
    def __init__(self, fpv_size, extra_size, hidden_size, cfg, num_layers=2):
        super(FusedLSTM, self).__init__()
        self.cfg = cfg
        num_classes = cfg.MODEL.NUM_CLASSES
        self.hidden_layer = hidden_size
        self.num_layers = num_layers

        self.use_mul_fuse = (cfg.model == 'm-lstm')
        self.use_extra = (cfg.use_extra)

        if self.use_extra:
            dim_in = fpv_size + extra_size
        else:
            dim_in = fpv_size

        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(dim_in, hidden_size)
        self.lstm = nn.LSTM(dim_in, hidden_size)

        self.fuse_lstm_cell = nn.LSTMCell(extra_size, extra_size)
        self.fuse_lstm = nn.LSTM(extra_size, extra_size)
        self.fuse_feat_ext = nn.Linear(extra_size, extra_size)

        self.summarize_hidden = nn.Linear(2 * hidden_size, hidden_size)
        self.summarize_cell = nn.Linear(2 * hidden_size, hidden_size)

        self.projection = nn.Linear(hidden_size, num_classes)
        self.task_feat = nn.Linear(hidden_size, hidden_size)
        self.task_projection = nn.Linear(hidden_size, len(Metadata.task))
        self.dropout = nn.Dropout(p=cfg.MODEL.DROPOUT_RATE)

    def forward(self, inputs, meta):
        fpv_features = inputs
        tpv_features = meta['extra_features']

        if self.use_extra:
            fused_tpv_features = []
            for pid_idx in range(inputs.size(2)):
                fused_tpv_features.append(self.fuse_feat_ext(tpv_features[:, :, pid_idx, :]).unsqueeze(2))
            fused_tpv_features = torch.cat(fused_tpv_features, dim=2)
            fused_tpv_features, _ = torch.max(fused_tpv_features, dim=2, keepdim=True)
            fused_tpv_features = fused_tpv_features.squeeze(0)
            fused_h, fused_c = self.fuse_lstm(fused_tpv_features)
            fused_h = fused_h.view(inputs.size(0), inputs.size(1), -1, fused_tpv_features.size(-1))
            fused_h = fused_h.repeat(1, 1, inputs.size(2), 1)
            input_feat = torch.cat((fpv_features, fused_h), dim=-1)
        else:
            input_feat = fpv_features

        input_feat = input_feat.squeeze(0)
        output, c = self.lstm(input_feat)
        output = output.unsqueeze(0)
        task_output = self.task_feat(output)
        output = self.projection(output)
        task_output = self.task_projection(task_output)
        return output, task_output


class MLP(nn.Module):
    def __init__(self, fpv_size, extra_size, cfg):
        super(MLP, self).__init__()
        num_classes = cfg.MODEL.NUM_CLASSES
        self.use_extra = cfg.use_extra

        if self.use_extra:
            dim = fpv_size + extra_size
        else:
            dim = fpv_size

        self.projection = nn.Linear(dim, num_classes)
        self.fuse_feat_ext = nn.Linear(extra_size, extra_size)
        self.fused_projection = nn.Linear(extra_size, extra_size)
        self.task_feat = nn.Linear(dim, dim)
        self.task_projection = nn.Linear(dim, len(Metadata.task))

    def forward(self, inputs, meta):
        fpv_features = inputs
        tpv_features = meta['extra_features']

        if self.use_extra:
            fused_tpv_features = []
            for pid_idx in range(inputs.size(2)):
                fused_tpv_features.append(self.fuse_feat_ext(tpv_features[0, :, pid_idx, :]).unsqueeze(1).unsqueeze(0))
            fused_tpv_features = torch.cat(fused_tpv_features, dim=2)
            fused_tpv_features, _ = torch.max(fused_tpv_features, dim=2, keepdim=True)
            fused_tpv_features = self.fused_projection(fused_tpv_features)
            fused_tpv_features = fused_tpv_features.repeat(1, 1, inputs.size(2), 1)
            input_feat = torch.cat((fpv_features, fused_tpv_features), dim=-1)
        else:
            input_feat = fpv_features

        task_output = self.task_feat(input_feat)
        output = self.projection(input_feat)
        task_output = self.task_projection(task_output)

        return output, task_output


class FeatureBank(nn.Module):
    def __init__(self, fpv_size, extra_size, cfg):
        super(FeatureBank, self).__init__()
        num_classes = cfg.MODEL.NUM_CLASSES
        self.use_extra = cfg.use_extra

        if self.use_extra:
            dim = 2 * (fpv_size + extra_size)
        else:
            dim = 2 * (fpv_size)

        self.projection = nn.Linear(dim, num_classes)
        self.fuse_feat_ext = nn.Linear(extra_size, extra_size)
        self.fused_projection = nn.Linear(extra_size, extra_size)
        self.task_feat = nn.Linear(dim, dim)
        self.task_projection = nn.Linear(dim, len(Metadata.task))
        self.window_length = 10

    def get_window(self, index):
        return [max(index - x - 1, 0) for x in range(self.window_length)]

    def forward(self, inputs, meta):
        fpv_features = inputs
        tpv_features = meta['extra_features']

        if self.use_extra:
            fused_tpv_features = []
            for pid_idx in range(inputs.size(2)):
                fused_tpv_features.append(self.fuse_feat_ext(tpv_features[0, :, pid_idx, :]).unsqueeze(1).unsqueeze(0))
            fused_tpv_features = torch.cat(fused_tpv_features, dim=2)
            fused_tpv_features, _ = torch.max(fused_tpv_features, dim=2, keepdim=True)
            fused_tpv_features = self.fused_projection(fused_tpv_features)
            fused_tpv_features = fused_tpv_features.repeat(1, 1, inputs.size(2), 1)
            input_feat = torch.cat((fpv_features, fused_tpv_features), dim=-1)
        else:
            input_feat = fpv_features

        long_term_features = []
        for sequence_idx in range(input_feat.size(1)):
            long_term_indices = self.get_window(sequence_idx)
            # Max pooling for long term feature bank
            long_term_feature, _ = torch.max(input_feat[0][long_term_indices], dim=0)
            long_term_feature = long_term_feature.unsqueeze(0)
            long_term_features.append(long_term_feature)

        long_term_features = torch.cat(long_term_features, dim=0).unsqueeze(0)
        input_feat = torch.cat((input_feat, long_term_features), dim=-1)
        task_output = self.task_feat(input_feat)
        output = self.projection(input_feat)
        task_output = self.task_projection(task_output)
        return output, task_output

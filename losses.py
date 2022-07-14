"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import random
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.t()).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.t()),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        same_cls = mask.sum(1)
        x = torch.ones(same_cls.shape[0], dtype=torch.float).to(device)
        same_cls = torch.where(same_cls == 0., x, same_cls)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / same_cls

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class infoNCE(nn.Module):
    def __init__(self, class_num):
        super(infoNCE, self).__init__()
        self.class_num = class_num
        self.neg_samples = 10

    def get_posAndneg(self, confi_labels_dict, feature_q_idx, source_centers, confi_class_idx):

        # get the label of q
        q_label = confi_labels_dict[feature_q_idx.item()]
        feature_pos = source_centers[q_label].unsqueeze(0)

        # get the negative samples
        negative_idx = []
        for i in range(self.class_num):
            if i != q_label:
                negative_idx.append(i)
        negative_pairs = torch.Tensor([]).cuda()
        for i in range(self.neg_samples):
            negative_pairs = torch.cat((negative_pairs, source_centers[random.choice(negative_idx)].unsqueeze(0)))
        # negative_pairs = torch.cat((source_centers[:feature_q_idx].unsqueeze(0), source_centers[feature_q_idx:].unsqueeze(0)))

        return torch.cat((feature_pos, negative_pairs))

    def reply_get_posAndneg(self, reply_label, source_centers):

        # get the label of q
        q_label = reply_label
        feature_pos = source_centers[q_label].unsqueeze(0)

        # get the negative samples
        negative_idx = []
        for i in range(self.class_num):
            if i != reply_label:
                negative_idx.append(i)
        negative_pairs = torch.Tensor([]).cuda()
        for i in range(self.neg_samples):
            negative_pairs = torch.cat((negative_pairs, source_centers[random.choice(negative_idx)].unsqueeze(0)))

        return torch.cat((feature_pos, negative_pairs))


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, lambda_num=3, beta=0.7):
        r"""Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """

        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.beta = beta
        self.lambda_num = lambda_num

    def forward(self, index, output, label, contrastive_loss=None):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """

        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (
                (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        ce_loss = F.cross_entropy(output, label)
        final_loss = ce_loss + self.lambda_num * elr_reg
        return final_loss

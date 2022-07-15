import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from sklearn.metrics import auc, confusion_matrix
import random
import os
import torch.utils.data
from torchvision import transforms
import numpy as np
from torch.utils import data
import torch.nn.functional as F
import sys
from scipy.spatial.distance import cdist
import argparse
import time
from torchvision import models
import copy
import torch.nn.utils.weight_norm as weightNorm
from losses import SupConLoss
from OH_datasets import FileListDataset
from os.path import join
from net.resnet import resnet18, resnet50, resnet101
from sklearn.model_selection import train_test_split


# from tensorboardX import SummaryWriter


def val_office(net, test_loader):
    net.eval()
    correct = 0
    total = 0

    gt_list = []
    p_list = []

    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
        gt_list.append(labels.cpu().numpy())
        with torch.no_grad():
            outputs, _ = net(inputs)
        # 取得分最高的那个类 (outputs.data的索引号)
        output_prob = F.softmax(outputs, dim=1).data
        p_list.append(output_prob[:, 1].detach().cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num

    acc = 100. * correct.item() / total

    return acc


def centers_val_office(net, test_loader, centers, confi_class_idx):
    net.eval()

    start_test = True
    with torch.no_grad():
        iter_test = iter(test_loader)
        final_centers = centers.cuda()

        for _ in range(len(test_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]

            inputs = inputs.cuda()
            _, feas = net(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        all_fea = all_fea.float().cpu().numpy()

        # validate the centers
        centers_output = net.fc(final_centers)
        output_prob = F.softmax(centers_output, dim=1).data
        _, predicted = torch.max(output_prob, 1)
        center_accuracy = torch.sum(
            torch.squeeze(predicted).float().cpu() == torch.tensor(confi_class_idx).float()).item() / len(
            confi_class_idx)

        final_centers = final_centers.float().cpu().numpy()
        dd = cdist(all_fea, final_centers, 'cosine')
        pred_label = dd.argmin(axis=1).tolist()
        confi_class_idx = np.array(confi_class_idx)
        prediction_c0 = confi_class_idx[pred_label]
        # change to real label index
        acc = np.sum(prediction_c0 == all_label.float().numpy()) / len(all_fea)

        print('target center acc is: %.3f, classification acc is: %.3f' % (center_accuracy, acc))

    return acc


def val_pclass(net, dataset_test, n_share, unk_class, top_ten_cls_idx):
    net.eval()

    correct = 0
    correct_close = 0
    size = 0
    # class_list = [i for i in range(n_share)]
    class_list = top_ten_cls_idx
    class_list.append(unk_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    per_class_correct_cls = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    all_gt = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0], data[1]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            out_t, _ = net(img_t)
            out_t = F.softmax(out_t)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            pred = out_t.data.max(1)[1]
            k = label_t.data.size()[0]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()

            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)

            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))
            size += k

    per_class_acc = np.round(per_class_correct / per_class_num, 3)

    return per_class_acc


def val_pclass_no_log(net, test_loader, logger, start, end):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(test_loader)
        for i in range(len(test_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs, _ = net(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal() / matrix.sum(axis=1) * 100

    avg_acc = 0.
    for i in range(start, end):
        avg_acc += acc[i]

    aacc = avg_acc / (end - start)
    # aacc = acc.mean()
    aa = [str(np.round(acc[i], 2)) for i in range(start, end)]
    acc = ' '.join(aa)
    logger.info('Acc Per-Class: {}, Acc Mean: {}'.format(acc, aacc))
    return aacc, acc


def val_pclass_OH(net, dataset_test, n_share, unk_class):
    net.eval()

    correct = 0
    correct_close = 0
    size = 0
    class_list = [i for i in range(n_share)]
    class_list.append(unk_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    per_class_correct_cls = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    all_gt = []
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t, label_t = data[0], data[1]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            out_t, _ = net(img_t)
            out_t = F.softmax(out_t)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            pred = out_t.data.max(1)[1]
            k = label_t.data.size()[0]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()

            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)

            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))
            size += k

    per_class_acc = np.round(per_class_correct / per_class_num, 3)

    return per_class_acc


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


# coding=utf-8

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.transforms as transforms
import argparse
import time
import torch.nn.functional as F
from OH_datasets import FileListDataset
from os.path import join
from net.resnet import resnet18, resnet50, resnet101


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='7', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--source', default=0, type=int)

    args = parser.parse_args()
    return args


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


def val_net(net, test_loader):
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
        output_prob = F.softmax(outputs, dim=1).data
        p_list.append(output_prob[:, 1].detach().cpu().numpy())
        _, predicted = torch.max(outputs, 1)
        total += inputs.size(0)
        num = (predicted == labels).sum()
        correct = correct + num

    acc = 100. * correct.item() / total

    return acc


class Trainer(object):
    def __init__(self):
        self.MSE_loss = nn.MSELoss().cuda()
        self.loss = nn.CrossEntropyLoss().cuda()

    def train_half(self, model, optimizer, x_val, y_val):
        model.train()
        output, _ = model(x_val)
        hloss = self.loss(output, y_val)

        optimizer.zero_grad()
        hloss.backward()
        optimizer.step()

        return hloss.item()

    def train(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        args = arg_parser()
        time_stamp_launch = time.strftime('%Y%m%d') + '-' + time.strftime('%H%M')
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        n_gpus = len(args.gpu.split(','))

        batch_size = args.batchsize
        epochs = args.max_epoch

        best_acc = 0.
        cls_nums = 65

        source_model_root = './model_source'
        if not os.path.exists(source_model_root):
            os.mkdir(source_model_root)

        net = resnet50(pretrained=True)
        net.fc = nn.Linear(2048, cls_nums)
        net = net.cuda()

        param_group = []
        for k, v in net.named_parameters():
            if k[:2] == 'fc':
                param_group += [{'params': v, 'lr': args.lr * 10}]
            else:
                param_group += [{'params': v, 'lr': args.lr}]

        optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4)

        # training dataset
        source = args.source
        source_classes = [i for i in range(cls_nums)]
        my_dataset = Dataset(
            path='../../dataset/office-home',
            domains=['Art', 'Clipart', 'Product', 'Real_World'],
            files=[
                'Art.txt',
                'Clipart.txt',
                'Product.txt',
                'Real_World.txt'
            ],
            prefix='../../dataset/office-home')
        source_domain = my_dataset.domains[source]
        source_file = my_dataset.files[source]
        print(
            'officehome_{}_all_train'.format(source_domain) + time_stamp_launch + 'model : resnet50  lr: %s' % args.lr)

        transform_train = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
        ])

        source_train_ds = FileListDataset(list_path=source_file, path_prefix=my_dataset.prefixes[source],
                                          transform=transform_train,
                                          filter=(lambda x: x in source_classes), return_id=False)

        train_loader = torch.utils.data.DataLoader(source_train_ds, batch_size=batch_size, shuffle=True,
                                                   num_workers=2 * n_gpus if n_gpus <= 2 else 2)

        source_val_ds = FileListDataset(list_path=source_file, path_prefix=my_dataset.prefixes[source],
                                        transform=transform_test,
                                        filter=(lambda x: x in source_classes), return_id=False)
        val_loader = torch.utils.data.DataLoader(source_val_ds, batch_size=batch_size, shuffle=False,
                                                 num_workers=2 * n_gpus if n_gpus <= 2 else 2)

        for i in range(epochs):
            running_loss = []
            net.train()

            for j, (img_data, img_label) in enumerate(train_loader):
                img_data = img_data.cuda()
                img_label = img_label.cuda()
                r_loss = self.train_half(net, optimizer, img_data, img_label)
                running_loss += [r_loss]
            avg_loss = np.mean(running_loss)
            if i % 3 == 0:
                acc = val_net(net, val_loader)
                print("Epoch %d running_loss=%.3f, acc=%.3f" % (i + 1, avg_loss, acc))

                if acc > best_acc:
                    best_acc = acc
                    torch.save(net,
                               './model_source/' + time_stamp_launch + '-OH_{}_ce_singe_gpu_resnet50_best.pkl'.format(
                                   my_dataset.domains[source]))
        print("Finished  Training")


if __name__ == '__main__':
    oct_trainer = Trainer()
    oct_trainer.train()

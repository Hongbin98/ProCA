import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from OH_datasets import FileListDataset, Imagenet_Dataset
from OH_datasets import my_Dataset as Dataset
import torch.nn.functional as F
from losses import infoNCE
from test import val_pclass
from tensorboardX import SummaryWriter
from PIL import Image
from test import val_office, centers_val_office
import sys
from net.resnet import resnet50


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='6', help='gpu device_ids for cuda')
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--max_epoch', default=15, type=int)
    parser.add_argument('--source_model', default='./model_source/20220714-1949-single_gpu_cal256_ce_resnet50_best.pkl')

    args = parser.parse_args()
    return args


class reply_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, buffer_per_class, soft_predictions):
        super(reply_dataset, self).__init__()

        start_cat = True
        for imgs in images:
            for img in imgs:
                if start_cat:
                    self.images = torch.tensor(img).unsqueeze(0)
                    start_cat = False
                else:
                    self.images = torch.cat((self.images, torch.tensor(img).unsqueeze(0)), dim=0)

        start_cat = True
        for label in labels:
            for i in range(buffer_per_class):
                if start_cat:
                    self.labels = torch.tensor(label).unsqueeze(0)
                    start_cat = False
                else:
                    self.labels = torch.cat((self.labels, torch.tensor(label).unsqueeze(0)))

        start_cat = True
        for soft_preds in soft_predictions:
            for soft_pred in soft_preds:
                if start_cat:
                    self.batch_soft_pred = soft_pred.unsqueeze(0)
                    start_cat = False
                else:
                    self.batch_soft_pred = torch.cat((self.batch_soft_pred, soft_pred.unsqueeze(0)), dim=0)

        self.images = self.images.cpu()
        self.labels = self.labels.cpu()
        self.batch_soft_pred = self.batch_soft_pred.cpu()

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.batch_soft_pred[index]

    def __len__(self):
        return self.labels.shape[0]


def obtain_label(loader, net, confi_class_idx):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            idx = data[2]
            inputs = inputs.cuda()
            outputs, feas = net(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_idx = idx.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_idx = torch.cat((all_idx, idx.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])

    # c_0 only get the confident-classes
    initc = initc[confi_class_idx]

    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1).tolist()
    confi_class_idx = np.array(confi_class_idx)
    prediction_c0 = confi_class_idx[pred_label]
    # change to real label index
    acc = np.sum(prediction_c0 == all_label.float().numpy()) / len(all_fea)

    # calculate c1 and pseudo-labels
    K = all_output.size(1)
    for round in range(1):
        aff = np.eye(K)[prediction_c0]

        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        # only get the confident-classes
        initc = initc[confi_class_idx]

        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        prediction_c1 = confi_class_idx[pred_label]
        acc = np.sum(prediction_c1 == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.3f}% -> {:.3f}%'.format(accuracy, acc)
    print(log_str + '\n')
    return dict(zip(all_idx.int().numpy(), prediction_c1)), accuracy * 100, acc * 100


def cosine_similarity(feature, pairs):
    feature = F.normalize(feature)
    pairs = F.normalize(pairs)
    similarity = feature.mm(pairs.t())
    return similarity


def get_confi_classes(source_model, target_data_loader, threshold=0.2):
    source_model.eval()
    prediction_bank = torch.zeros(1, 256).cuda()
    for j, (img_data, _, _) in enumerate(target_data_loader):
        img_data = img_data.cuda()
        with torch.no_grad():
            output, _ = source_model(img_data)
        output_prob = F.softmax(output, dim=1).data
        batch_prob_sum = torch.sum(output_prob, dim=0)
        prediction_bank += batch_prob_sum

    confi_class_idx = []
    sort_bank, sort_class_idx = torch.sort(prediction_bank, descending=True)

    # min max scaler
    sort_bank = sort_bank.squeeze(0)
    prediction_bank = prediction_bank.squeeze(0)
    max_cls = sort_bank[0]
    min_cls = sort_bank[-1]
    for idx, value in enumerate(prediction_bank):
        prediction_bank[idx] = (prediction_bank[idx] - min_cls) / (max_cls - min_cls)

    confuse_cls_idx = []

    for idx, value in enumerate(prediction_bank):
        if value >= threshold:
            confi_class_idx.append(idx)

    return confi_class_idx, confuse_cls_idx, prediction_bank[confi_class_idx]


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


def get_source_centers(source_loader, net):
    net.eval()

    fea_list = []
    label_list = []
    with torch.no_grad():
        iter_test = iter(source_loader)
        for _ in tqdm(range(len(source_loader))):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs, feas = net(inputs)

            fea_list.append(feas.float().cpu())
            label_list.append(labels)

    K = outputs.size(1)
    all_label = torch.cat(label_list, 0)
    all_fea = torch.cat(fea_list, 0)

    total_source_protos = torch.empty(K, all_fea.size(1), dtype=all_fea.dtype)
    for i in range(K):
        total_source_protos[i] = all_fea[all_label == i].mean(dim=0)

    return total_source_protos.cuda()


def get_one_classes_imgs(target_train_loader, class_idx, confi_label_dict):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(target_train_loader)
        for _ in range(len(target_train_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            sample_idx = data[2]
            if start_test:
                all_inputs = inputs.float().cpu()
                all_idx = sample_idx.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.float().cpu()), 0)
                all_idx = torch.cat((all_idx, sample_idx.float()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

        print('construct class %s examplar.' % (class_idx))
        imgs_idx = []
        for cnt_idx, idx in enumerate(all_idx):
            if int(idx.item()) in confi_label_dict:
                if confi_label_dict[int(idx.item())] == class_idx:
                    imgs_idx.append(cnt_idx)

        return all_inputs[imgs_idx]


def get_buffer_centers(reply_loader, net, confi_class_total_idx):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(reply_loader)
        for _ in range(len(reply_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            _ = data[2]

            inputs = inputs.cuda()
            _, feas = net(inputs)

            if start_test:
                all_fea = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # for each class
    all_fea = all_fea.float().cpu()
    # for each class
    for idx, cls in enumerate(confi_class_total_idx):
        cnt = 0
        for i, label in enumerate(all_label):
            if label == cls:
                if cnt == 0:
                    target_cls_proto = all_fea[i]
                else:
                    target_cls_proto += all_fea[i]

                cnt += 1

        target_cls_proto = target_cls_proto / cnt

        if idx == 0:
            total_target_protos = target_cls_proto.unsqueeze(0)
        else:
            total_target_protos = torch.cat((total_target_protos, target_cls_proto.unsqueeze(0)), 0)

    return total_target_protos


def get_target_centers(target_train_loader, net, confi_class_idx, confi_label_dict):
    net.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(target_train_loader)
        for _ in range(len(target_train_loader)):
            data = iter_test.next()
            inputs = data[0]
            sample_idx = data[2]
            inputs = inputs.cuda()
            _, feas = net(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_idx = sample_idx.float().cpu()
                start_test = False
            else:
                all_idx = torch.cat((all_idx, sample_idx.float()), 0)
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)

    # for each class
    for idx, cls in enumerate(confi_class_idx):
        cnt = 0
        for i, sam_idx in enumerate(all_idx):
            if confi_label_dict[sam_idx.item()] == cls:
                if cnt == 0:
                    target_cls_proto = all_fea[i]
                else:
                    target_cls_proto += all_fea[i]

                cnt += 1

        target_cls_proto = target_cls_proto / cnt

        if idx == 0:
            total_target_protos = target_cls_proto.unsqueeze(0)
        else:
            total_target_protos = torch.cat((total_target_protos, target_cls_proto.unsqueeze(0)), 0)

    return total_target_protos


#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
"""
So the reply buffer should 3 functions
1. store the reply buffers(original data) and their soft-predictions to prevent catastrophic forgetting
2. combine the target data with the reply buffers, and calculate the features to align both domains
3. calculate the prototype(mean features) of each class to do classification
"""


class reply_buffer():
    def __init__(self, transform, imgs_per_class=20):
        super(reply_buffer, self).__init__()
        self.exemplar_set = []
        self.soft_pred = []
        self.target_center_set = []
        self.transform = transform
        self.m = imgs_per_class

    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, model, images, transform):
        model.eval()
        with torch.no_grad():
            x = images.cuda()
            model = model
            output, feas = model(x)
            feature_extractor_output = F.normalize(feas.detach()).cpu().numpy()
            class_mean = np.mean(feature_extractor_output, axis=0)

            class_center = np.mean(feas.detach().cpu().numpy(), axis=0)

            # get the probability
            output = nn.Softmax(dim=1)(output)

        return class_mean, feature_extractor_output, output, class_center

    def construct_exemplar_set(self, images, model):
        class_mean, feature_extractor_output, buffer_output, class_center = self.compute_class_mean(model, images,
                                                                                                    self.transform)
        exemplar = []
        soft_predar = []
        feas_past = []
        now_class_mean = np.zeros((1, 2048))  # for ResNet-50

        for i in range(self.m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]

            exemplar.append(images[index])
            soft_predar.append(buffer_output[index])
            feas_past.append(feature_extractor_output[index])

        self.exemplar_set.append(exemplar)
        self.soft_pred.append(soft_predar)

    def update_exemplar_set(self, images, model, history_idx):
        class_mean, feature_extractor_output, buffer_output, class_center = self.compute_class_mean(model, images,
                                                                                                    self.transform)
        exemplar = []
        soft_predar = []
        feas_past = []
        now_class_mean = np.zeros((1, 2048))  # for ResNet-50

        for i in range(self.m):
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(images[index])
            soft_predar.append(buffer_output[index])
            feas_past.append(feature_extractor_output[index])

        self.exemplar_set[history_idx] = exemplar
        self.soft_pred[history_idx] = soft_predar


if __name__ == '__main__':
    args = arg_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_sharing_strategy('file_system')
    writer = SummaryWriter()

    last_acc_stages = []
    best_acc_stages = []
    center_acc_stages = []
    top_ten_stages = []

    total_cls_nums = 256
    incre_cls_nums = 10
    reply_buffer_nums = 10
    batch_size = args.batchsize
    pseudo_update_interval = 10
    prototypes_update_interval = 10
    source_centers_update_interval = 10

    diff_of_centers = torch.zeros((total_cls_nums, 2048))

    # optimizer
    lr = args.lr
    weight_decay = 1e-6
    momentum = 0.9
    n_epoches = args.max_epoch

    # dataset
    cal_dataset = Dataset(
        path='../../dataset/ImageNet-Caltech',
        domains=['256_ObjectCategories'],
        files=[
            'caltech_list.txt',
        ],
        prefix='../../dataset/ImageNet-Caltech')

    imgNet_dataset = Imagenet_Dataset(
        path='../../dataset/ImageNet-Caltech',
        domains=['imageNet_84_val'],
        files=[
            'imagenet_84_list.txt',
        ],
        prefix='../../dataset/ImageNet-Caltech')

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # grayscale mean/std
    ])

    # loss functions
    margin = 0.3
    gamma = 0.07
    info_nce = infoNCE(class_num=total_cls_nums).cuda()
    nll = nn.NLLLoss()
    ce_loss = nn.CrossEntropyLoss()
    contrastive_label = torch.tensor([0]).cuda()

    current_step = 0
    confi_cls_history = []
    confi_cls_value = np.zeros(total_cls_nums)
    reply_buffer = reply_buffer(transform_test, reply_buffer_nums)

    source = 0
    target = 0

    source_file = cal_dataset.files[0]
    target_file = imgNet_dataset.files[0]

    # pre-trained model
    pretrained_net = torch.load(args.source_model)
    pretrained_net = pretrained_net.cuda()
    pretrained_net.eval()

    cal_84_cls_list = [188, 0, 165, 145, 33, 94, 76, 86, 71, 128, 225, 108, 219, 115, 87, 44, 62, 163, 85, 177, 134,
                       229, 82, 60, 29, 172, 114, 160, 146, 228, 245, 75, 157, 209, 97, 106, 170, 198, 92, 230, 234, 40,
                       200, 151, 11, 27, 185, 47, 45, 9, 88, 37, 30, 2, 90, 196, 181, 28, 253, 112, 178, 109, 39, 96,
                       192, 110, 211, 237, 249, 89, 215, 116, 126, 133, 107, 123, 193, 68, 179, 227, 150, 141, 7, 50]
    cal_84_cls_list.sort()

    top_ten_cls_idx = cal_84_cls_list[:10]
    for incre_idx in range(len(cal_84_cls_list) // incre_cls_nums):  # only for the first stage
        source_total_classes = [i for i in range(total_cls_nums)]
        target_train_classes = [cal_84_cls_list[i] for i in
                                range(incre_cls_nums * incre_idx, incre_cls_nums * incre_idx + incre_cls_nums)]
        target_test_classes = [cal_84_cls_list[i] for i in range(0, incre_cls_nums * incre_idx + incre_cls_nums)]

        source_total_ds = FileListDataset(list_path=source_file, path_prefix=cal_dataset.prefixes[source],
                                          transform=transform_test,
                                          filter=(lambda x: x in source_total_classes), return_id=False)
        source_total_loader = torch.utils.data.DataLoader(source_total_ds, batch_size=batch_size,
                                                          shuffle=True,
                                                          num_workers=2 * 4)

        target_train_ds = FileListDataset(list_path=target_file, path_prefix=imgNet_dataset.prefixes[target],
                                          return_id=True,
                                          transform=transform_test,
                                          filter=(lambda x: x in target_train_classes))

        target_train_dl = DataLoader(dataset=target_train_ds, batch_size=batch_size, shuffle=True,
                                     num_workers=2 * 4)

        target_test_ds = FileListDataset(list_path=target_file, path_prefix=imgNet_dataset.prefixes[target],
                                         transform=transform_test,
                                         filter=(lambda x: x in target_test_classes),
                                         return_id=False)
        target_test_loader = torch.utils.data.DataLoader(target_test_ds, batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=2 * 4)

        confi_class_idx, confuse_cls_idx, confi_class_values = get_confi_classes(pretrained_net,
                                                                                 target_train_dl,
                                                                                 threshold=0.15)
        print(target_train_classes)
        print(confi_class_idx)
        print('From {} to {}, the source-only accuracy is:'.format(cal_dataset.domains[source],
                                                                   imgNet_dataset.domains[target]))
        pred_label_dict, _, _ = obtain_label(target_train_dl, pretrained_net, confi_class_idx)

        if incre_idx == 0:
            net = torch.load(args.source_model)
            net = net.cuda()
        else:
            net = torch.load('./model_source/C2I_{}_2_{}_Resnet50_DA_last_stage{}.pt'.format(source, target, incre_idx - 1))

        # optimizer
        param_group = []
        for p in net.parameters():
            p.requires_grad = True
        for k, v in net.named_parameters():
            if k[:2] == 'fc':
                param_group += [{'params': v, 'lr': lr}]
            else:
                param_group += [{'params': v, 'lr': lr}]
        optimizer = optim.SGD(param_group, momentum=momentum, weight_decay=weight_decay)
        best_tar_acc = 0.
        this_stage_save_imgs = True

        for epoch in range(n_epoches):

            iter_target_train = iter(target_train_dl)
            iter_source_train = iter(source_total_loader)

            min_iterations = min(len(target_train_dl), len(source_total_loader))

            if reply_buffer.exemplar_set:  # if there are any prototype-images
                # get the reply buffer loader
                reply_ds = reply_dataset(images=reply_buffer.exemplar_set, labels=confi_cls_history,
                                         buffer_per_class=reply_buffer_nums,
                                         soft_predictions=reply_buffer.soft_pred)
                reply_loader = torch.utils.data.DataLoader(reply_ds, batch_size=batch_size, shuffle=True)
                iter_reply_buffer = iter(reply_loader)

            if epoch % pseudo_update_interval == 0 and epoch != 0:
                pred_label_dict, _, _ = obtain_label(target_train_dl, pretrained_net, confi_class_idx)

            sum_contras = torch.tensor(0.).cuda()
            for iter_idx in range(min_iterations):
                optimizer.zero_grad()

                net.train()

                #########################################################
                # Source data cross-entropy optimization
                source_data = iter_source_train.next()
                source_inputs = source_data[0].cuda()
                source_labels = source_data[1].cuda()

                source_ouputs, _ = net(source_inputs)
                source_ce = ce_loss(source_ouputs, source_labels)
                #########################################################

                #########################################################
                # Target data optimization
                data = iter_target_train.next()
                inputs = data[0]
                ground_truths = data[1].cuda()
                sample_idx = data[2]
                inputs = inputs.cuda()

                ouputs, feas = net(inputs)

                ce_sample_idx = sample_idx.numpy().tolist()
                pseudo_labels = []
                for each_idx in ce_sample_idx:
                    pseudo_labels.append(pred_label_dict[each_idx])

                pseudo_labels = torch.tensor(pseudo_labels).cuda()
                loss_ce = ce_loss(ouputs, pseudo_labels)
                #########################################################

                #########################################################
                # target prototype contrastive alignment(for DA)  and  reply buffers distillation
                tar_contras_loss = torch.tensor(0.).cuda()
                distill_loss = torch.tensor(0.).cuda()
                if reply_buffer.exemplar_set:  # if there are any prototype-images
                    data_buffer = next(iter_reply_buffer, -1)
                    if data_buffer == -1:
                        data_target_iter = iter(reply_loader)
                        re_org_img, re_org_label, re_org_sp = data_target_iter.next()
                    else:
                        re_org_img, re_org_label, re_org_sp = data_buffer

                    re_org_img = re_org_img.cuda()
                    re_org_label = re_org_label.cuda()
                    re_org_sp = re_org_sp.cuda()

                    reply_ouputs, reply_feas = net(re_org_img)

                    # contrastive alignment
                    reply_con_loss = torch.tensor(0.).cuda()
                    for idx, fea in enumerate(reply_feas):
                        pos_neg_pair = info_nce.reply_get_posAndneg(re_org_label[idx], source_centers)
                        result = cosine_similarity(fea.unsqueeze(0), pos_neg_pair)
                        numerator = torch.exp((result[0][0] - margin) / gamma)
                        denominator = numerator + torch.sum(torch.exp((result / gamma)[0][1:]))
                        result = torch.log(numerator / denominator).unsqueeze(0).unsqueeze(0)
                        contrastive_loss = nll(result, contrastive_label)
                        reply_con_loss += contrastive_loss
                    reply_con_loss /= len(reply_feas)
                    tar_contras_loss += reply_con_loss
                    #########################################################
                    # distillation
                    # calculate the sf-pred cross-entropy
                    reply_ouputs = nn.Softmax(dim=1)(reply_ouputs)  # get the softmax-output
                    reply_ouputs = torch.log(reply_ouputs)  # get the log-softmax
                    soft_pred_loss = torch.sum(-1 * re_org_sp * reply_ouputs, dim=1)  # -1 * p(x) * log q(x)
                    soft_pred_loss = torch.mean(soft_pred_loss)

                    distill_loss += soft_pred_loss
                #########################################################
                # source_ce and loss_ce are CE part
                # tar_contras_loss are contrastive part
                # loss1 is the distillation part
                total_loss = (source_ce + loss_ce) + 0.1 * tar_contras_loss + 1.0 * distill_loss
                total_loss.backward()
                optimizer.step()

                sum_contras += tar_contras_loss

            if epoch == 3 or (
                    epoch != 0 and epoch % prototypes_update_interval == 0):  # after the warm-up stage, update the imgs 1 time / per 3 epoches
                for confi_idx, confi_class in enumerate(confi_class_idx):
                    if this_stage_save_imgs:
                        if confi_class not in confi_cls_history:
                            confi_cls_history.append(confi_class)
                            confi_cls_value[confi_class] = confi_class_values[confi_idx]

                            imgs = get_one_classes_imgs(target_train_dl, confi_class, pred_label_dict)
                            reply_buffer.construct_exemplar_set(imgs, net)
                    else:
                        history_idx = confi_cls_history.index(confi_class)
                        if confi_class_values[confi_idx] >= confi_cls_value[confi_class]:
                            imgs = get_one_classes_imgs(target_train_dl, confi_class, pred_label_dict)
                            reply_buffer.update_exemplar_set(imgs, net, history_idx)

                this_stage_save_imgs = False

                print('Source-centers update')
                source_centers = get_source_centers(source_total_loader, pretrained_net)
                print('Finish!')

            acc_list = val_pclass(net, target_test_loader, total_cls_nums, total_cls_nums, target_test_classes)
            if incre_idx < total_cls_nums // incre_cls_nums:
                acc_list = acc_list[:(incre_idx + 1) * incre_cls_nums]

            if incre_cls_nums >= 10:
                top_ten_classes_mean = acc_list[:10]

            avg_contras = sum_contras / min_iterations
            print('Epoch: %d, source ce_loss is %.3f, target pseudo_label ce loss is %.3f' % (
                epoch, source_ce.item(), loss_ce.item()))
            print('Epoch: %d, target_prototypes contrastive loss is %.3f' % (
                epoch, avg_contras.item()))
            print('Epoch: %d, distillation loss is %.3f' % (
                epoch, distill_loss.item()))
            print('top ten classes mean acc is %.3f' % top_ten_classes_mean.mean())

            # save the tensorboard logs
            current_step += 1
            writer.add_scalar('source ce_loss', source_ce.item(), global_step=current_step)
            writer.add_scalar('target pseudo_label ce loss', loss_ce.item(), global_step=current_step)
            # writer.add_scalar('pseudo labels contrastive loss', tar_con_loss.item(), global_step=current_step)
            writer.add_scalar('buffer contrastive loss', avg_contras.item(), global_step=current_step)
            writer.add_scalar('distillation loss', distill_loss.item(), global_step=current_step)

            writer.add_scalar('total_per_class_acc', acc_list.mean(), global_step=current_step)
            _, source_only_acc, cluster_acc = obtain_label(target_train_dl, net, confi_class_idx)
            writer.add_scalar('stage_target_train_acc', source_only_acc, global_step=current_step)
            writer.add_scalar('cluster_acc', cluster_acc, global_step=current_step)

            # get the whole loader acc
            total_mean_acc = val_office(net, target_test_loader)
            writer.add_scalar('total_mean_acc', total_mean_acc, global_step=current_step)
            print('total_mean_acc is %.3f' % total_mean_acc)

            if total_mean_acc > best_tar_acc:
                best_tar_acc = total_mean_acc
                # torch.save(net, './model_source/{}_2_{}_Resnet50_DA_Best_stage{}.pt'.format(source, target, incre_idx))

            torch.save(net, './model_source/C2I_{}_2_{}_Resnet50_DA_last_stage{}.pt'.format(source, target, incre_idx))
        best_acc_stages.append(best_tar_acc)
        ################################################################################################
        # centers classification
        # after training, get the target centers
        stage_target_protos = get_buffer_centers(reply_loader, net, confi_class_idx)
        stage_tar_centers = get_target_centers(target_train_dl, net, confi_class_idx, pred_label_dict)

        center_diff = stage_tar_centers - stage_target_protos
        for i, cls in enumerate(confi_class_idx):
            diff_of_centers[cls] = center_diff[i]

        final_protos = get_buffer_centers(reply_loader, net, confi_cls_history)
        for i, cls in enumerate(confi_cls_history):
            final_protos[i] += diff_of_centers[cls]
        center_acc = centers_val_office(net, target_test_loader, final_protos, confi_cls_history)
        print('Proto-centers acc is: %.3f' % center_acc)
        center_acc_stages.append(center_acc)
        ################################################################################################
        # last acc & top ten classes acc
        last_total_mean_acc = val_office(net, target_test_loader)
        last_acc_stages.append(last_total_mean_acc / 100)
        top_ten_stages.append(top_ten_classes_mean.mean())

    print('From {} to {}, the best accuracy of different stages is:'.format(imgNet_dataset.domains[source],
                                                                            cal_dataset.domains[target]))
    print(np.round(best_acc_stages, 3))

    print('From {} to {}, the last accuracy of different stages is:'.format(imgNet_dataset.domains[source],
                                                                            cal_dataset.domains[target]))
    print(np.round(last_acc_stages, 3))

    print('From {} to {}, the top_ten accuracy of different stages is:'.format(imgNet_dataset.domains[source],
                                                                               cal_dataset.domains[target]))
    print(np.round(top_ten_stages, 3))

    print('From {} to {}, the center accuracy of different stages is:'.format(imgNet_dataset.domains[source],
                                                                              cal_dataset.domains[target]))
    print(np.round(center_acc_stages, 3))

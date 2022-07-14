import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class my_Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(os.path.join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)

class Imagenet_Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(os.path.join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


def join_path(*a):
    return os.path.join(*a)


def one_hot(class_ids, num_classes):
    labels = torch.tensor(class_ids).view(-1, 1)
    batch_size = labels.numel()
    return torch.zeros(batch_size, num_classes, dtype=torch.float, device=labels.device).scatter_(1, labels, 1)


class TestDataset(Dataset):
    """
    simple test dataset to store N data, ith data is ``([i, i+1], [2i+1])`` where 0 <= i < N
    """

    def __init__(self, N=100):
        self.N = N

    def __getitem__(self, index):
        return [index, index + 1], [2 * index + 1]

    def __len__(self):
        return self.N


class BaseImageDataset(Dataset):
    """
    base image dataset

    for image dataset, ``__getitem__`` usually reads an image from a given file path

    the image is guaranteed to be in **RGB** mode

    subclasses should fill ``datas`` and ``labels`` as they need.
    """

    def __init__(self, transform=None, return_id=False):
        self.return_id = return_id
        self.transform = transform or (lambda x: x)
        self.datas = []
        self.labels = []

    def __getitem__(self, index):
        im = Image.open(self.datas[index]).convert('RGB')
        im = self.transform(im)
        if not self.return_id:
            return im, self.labels[index]
        return im, self.labels[index], index

    def __len__(self):
        return len(self.datas)


class FileListDataset(BaseImageDataset):
    """
    dataset that consists of a file which has the structure of :

    image_path label_id
    image_path label_id
    ......

    i.e., each line contains an image path and a label id
    """

    def __init__(self, list_path, path_prefix='', transform=None, return_id=False, num_classes=None, filter=None):
        """
        :param str list_path: absolute path of image list file (which contains (path, label_id) in each line) **avoid space in path!**
        :param str path_prefix: prefix to add to each line in image list to get the absolute path of image,
            esp, you should set path_prefix if file path in image list file is relative path
        :param int num_classes: if not specified, ``max(labels) + 1`` is used
        :param int -> bool filter: filter out the data to be used
        """
        super(FileListDataset, self).__init__(transform=transform, return_id=return_id)
        self.list_path = list_path
        self.path_prefix = path_prefix
        filter = filter or (lambda x: True)

        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines() if
                    line.strip()]  # avoid empty lines
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is a space in the image path?')
                raise e

        ans = [(x, y) for (x, y) in zip(self.datas, self.labels) if filter(y)]
        self.datas, self.labels = zip(*ans)

        self.num_classes = num_classes or max(self.labels) + 1


class UnLabeledImageDataset(BaseImageDataset):
    """
    applies to image dataset in one directory without labels for unsupervised learning, like getchu, celeba etc

    **although this is UnLabeledImageDataset, it returns useless labels to have similar interface with other datasets**
    """

    def __init__(self, root_dir, transform=None, return_id=False):
        """

        :param root_dir:  search ``root_dir`` recursively for all files (treat all files as image files)
        """
        super(UnLabeledImageDataset, self).__init__(transform=transform, return_id=return_id)
        self.root_dir = root_dir
        self.datas = sum(
            [[os.path.join(path, file) for file in files] for path, dirs, files in os.walk(self.root_dir) if files], [])
        self.labels = [0 for x in self.datas]  # useless label

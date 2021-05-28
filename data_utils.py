import os, operator, random, wget, tarfile
from torch.utils.data import Dataset
import torch
from PIL import Image
import matplotlib.pyplot as plt


# dataset object
class Cifar50Dataset(Dataset):
    def __init__(self, cifar_json, transform):
        self.class_num = cifar_json['num_classes']
        self.annotations = cifar_json['annotations']
        self.transform = transform

        self.label_dict = {}
        for idx, data_ref in enumerate(self.annotations):
            label = data_ref['category']
            label_id = data_ref['category_id']
            if label_id not in self.label_dict:
                self.label_dict[label_id] = label

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations[idx]['fpath']
        label_id = self.annotations[idx]['category_id']
        # label = self.annotations[idx]['category']

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        return image, label_id


# utils for original cifar100 dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def arr2tensor(arr):
    assert len(arr) == 3072
    return torch.tensor([arr[:1024].reshape(32, 32),
                        arr[1024:2048].reshape(32, 32),
                        arr[2048:3072].reshape(32, 32)])


def get_cifar100_labels():
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = os.path.basename(url)
    if not os.path.exists(filename):
        filename = wget.download(url)
    assert filename.endswith("tar.gz"), "tar file url error"
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()
    basename = os.path.splitext(os.path.splitext(filename)[0])[0]
    meta = unpickle(os.path.join(basename, "meta"))
    fine_labels = [n.decode("utf-8") for n in meta[b'fine_label_names']]
    coarse_labels = [n.decode("utf-8") for n in meta[b'coarse_label_names']]

    return fine_labels, coarse_labels


def print_random_cifar100_test(fine_labels, coarse_labels):
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = os.path.basename(url)
    basename = os.path.splitext(os.path.splitext(filename)[0])[0]
    test = unpickle(os.path.join(basename, "test"))
    rand_idx = random.randint(0, len(test[b'data'])-1)
    fine_label = fine_labels[test[b'fine_labels'][rand_idx]]
    coarse_label = coarse_labels[test[b'coarse_labels'][rand_idx]]
    img = arr2tensor(test[b'data'][rand_idx])
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"{fine_label} {coarse_label}")


# Other dataset utils
def print_random_img(dataset, label_map, n=3):
    fig, axes = plt.subplots(1, n, figsize=(8, 8*n))
    for i in range(n):
        rand_idx = random.randint(0, len(dataset))
        img, target = dataset[rand_idx]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(label_map[target])


def print_random_img_by_dataset(dataset_arr, label_map, n=3):
    rand_idx_arr = [random.randint(0, len(dataset_arr[0])) for i in range(n)]
    for idx, dataset in enumerate(dataset_arr):
        fig, axes = plt.subplots(1, n, figsize=(8, 8*n))
        for i in range(n):
            img, target = dataset[rand_idx_arr[i]]
            axes[i].imshow(img.permute(1, 2, 0))
            axes[i].set_title(label_map[target])
        fig.show()


def generate_jsonDict(main_json):
    data_ref_arr = main_json["annotations"]
    class_num = main_json["num_classes"]

    json_dict  = {}
    label_dict = {}

    for idx, data_ref in enumerate(data_ref_arr):
        label = data_ref['category']
        label_id = data_ref['category_id']
        if label_id not in label_dict:
            label_dict[label_id] = label
            json_dict[label_id] = {"annotations": [], "num_classes": 1}
        json_dict[label_id]["annotations"].append(data_ref)

    return json_dict, label_dict


def plot_distribution(json_dict, label_dict, decending=True):
    label_to_length = {k: len(json_dict[k]['annotations']) for k in json_dict.keys()}

    sorted_tuples = sorted(label_to_length.items(),
                           key=operator.itemgetter(1),
                           reverse=decending)
    sorted_dict = {k: v for k, v in sorted_tuples}

    width = 1.0
    plt.figure(figsize=(20, 7))  # width:20, height:10
    # [ label_dict[k]  for k in sorted_dict.keys()]
    plt.bar([label_dict[k] for k in sorted_dict.keys()],
            sorted_dict.values(), width, color='g', align='center')
    plt.xticks(rotation=90)
    plt.show()
    return sorted_dict

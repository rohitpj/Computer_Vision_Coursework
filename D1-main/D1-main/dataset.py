# TODO: Implement the dataset class extending torch.utils.data.Dataset
import os
import pickle
import torch
import torchvision.datasets
import torchvision.transforms as T

class ModifiedCIFAR100(torch.utils.data.Dataset):
    """ CIFAR100 dataset with augment. """
    def __init__(self, root='./data/', train=True, augment=False):
        self.root = root
        self.augment = augment and train

        if train:
            data = pickle.load(open(os.path.join(root, 'train.pkl'), 'rb'), encoding='bytes')
        else:
            data = pickle.load(open(os.path.join(root, 'test.pkl'), 'rb'), encoding='bytes')

        self.fine_labels = data[b'fine_labels']
        self.coarse_labels = data[b'coarse_labels']
        self.data = data[b'data']

        # Standard ImageNet normalization
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        if self.augment:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.RandomCrop(224, padding=28),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            # Validation / test transforms (just preprocessing)
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        coarse_label = self.coarse_labels[index]
        fine_label = self.fine_labels[index]

        img = img.reshape(3, 32, 32).transpose(1, 2, 0)
        img = self.transform(img)

        return {'image': img, 'coarse_label': coarse_label, 'fine_label': fine_label}

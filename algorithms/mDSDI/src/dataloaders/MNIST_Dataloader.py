import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MNISTDataloader(Dataset):
    def __init__(self, src_path, sample_paths, class_labels, domain_label=-1):
        self.image_transformer = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
        self.src_path = src_path
        self.domain_label = domain_label
        self.sample_paths, self.class_labels = sample_paths, class_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path)
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.src_path + self.sample_paths[index])
        class_label = self.class_labels[index]

        return sample, class_label, self.domain_label


class MNIST_Test_Dataloader(MNISTDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self.image_transformer = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])

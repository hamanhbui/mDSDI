from algorithms.ERM.src.dataloaders.MNIST_Dataloader import MNIST_Test_Dataloader, MNISTDataloader
from algorithms.ERM.src.dataloaders.Standard_Dataloader import StandardDataloader, StandardValDataloader


train_dataloaders_map = {
    "PACS": StandardDataloader,
    "DomainNet": StandardDataloader,
    "MNIST": MNISTDataloader,
    "OfficeHome": StandardDataloader,
    "VLCS": StandardDataloader,
}

test_dataloaders_map = {
    "PACS": StandardValDataloader,
    "DomainNet": StandardValDataloader,
    "MNIST": MNIST_Test_Dataloader,
    "OfficeHome": StandardValDataloader,
    "VLCS": StandardValDataloader,
}


def get_train_dataloader(name):
    if name not in train_dataloaders_map:
        raise ValueError("Name of train dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return train_dataloaders_map[name](**kwargs)

    return get_dataloader_fn


def get_test_dataloader(name):
    if name not in test_dataloaders_map:
        raise ValueError("Name of test dataloader unknown %s" % name)

    def get_dataloader_fn(**kwargs):
        return test_dataloaders_map[name](**kwargs)

    return get_dataloader_fn

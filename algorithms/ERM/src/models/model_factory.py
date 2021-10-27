from algorithms.ERM.src.models.mnistnet import MNIST_CNN, Color_MNIST_CNN
from algorithms.ERM.src.models.resnet import ResNet


nets_map = {"mnistnet": MNIST_CNN, "cmnistnet": Color_MNIST_CNN, "resnet50": ResNet}


def get_model(name):
    if name not in nets_map:
        raise ValueError("Name of model unknown %s" % name)

    def get_model_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_model_fn

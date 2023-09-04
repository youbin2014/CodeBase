import torch.nn as nn
from Models.resnet import get_resnet
from Models.vgg import get_vgg
def get_model(model_name, pretrained=False, num_classes=10,normalize=None,dataset="cifar10"):
    model_name=model_name.lower()
    if "resnet" in model_name:
        model= get_resnet(model_name, pretrained, num_classes,dataset=dataset)
    elif "vgg" in model_name:
        model= get_vgg(model_name, pretrained, num_classes,dataset=dataset)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    if normalize:
        model = nn.Sequential(normalize, model)
    return model
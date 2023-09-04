import torchvision.models as models
import torch


def get_vgg(pretrained=False, num_classes=10):
    """
    Returns a VGG16 model.

    Modify the last layer to fit the dataset if not ImageNet pretrained model is downloaded


    Parameters:
    - pretrained (bool): Whether to load pretrained weights.
    - num_classes (int): Number of output classes.

    Returns:
    - model (torch.nn.Module): VGG model.
    """
    model = models.vgg16(weights=pretrained)
    if not pretrained:
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
    return model

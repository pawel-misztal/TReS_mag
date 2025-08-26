from torchinfo import summary

import os
import socket
hostname = socket.gethostname()
if hostname == 'mrpawlinux': 
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'


def printModelData(model):
  print(summary(
    model=model,
    input_size=(1,3,224,224),
    col_names=["input_size", "output_size", "num_params", "trainable"],
    row_settings=['var_names']))
  

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, efficientnet_b4, EfficientNet_B4_Weights, resnet50, ResNet50_Weights

effnet = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
effnet.cpu()

convn = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
convn.cpu()

resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
resnet.cpu()

printModelData(effnet)
printModelData(convn)
printModelData(resnet)
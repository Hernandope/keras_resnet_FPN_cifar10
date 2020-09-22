# Keras-based Feature Pyramid Network model customized for cifar10

Feature Pyramid Network is ubiquitious for detection, semantic segmentation, or even panoptic segmentation, still found in state-of-art models.
Most implementations around are done in PyTorch, here I made my own spin on FPN in keras for usage in TF-based projects.

Main objective of creating this project is to compare FPN performance with different backbones for object detection.
Main code can be found in my jupyter notebook trained and ran.
Please check `new_model.py`for how the model is defined.

## Backbones compatibility
Bottom-up pathway in my implementation is compatible with:
*ResNet50 
*ResNet101
*ResNet50V2
*ResNet101V2

Feel free to add compatibility with other backbones!
Possible backbones to add:
*Wide ResNet
*ResNeSt
*ResNeXt
*Models with Res2Net

## References:
Original FPN paper: https://arxiv.org/abs/1612.03144
Panoptic FPN paper: https://arxiv.org/abs/1901.02446
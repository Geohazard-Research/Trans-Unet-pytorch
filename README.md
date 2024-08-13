# Trans-Unet Model for Landslide Mapping
The overall architecture of the Trans-UNet2 model, as depicted in Figure 1, employs a symmetric encoder-decoder structure similar to that of the UNet model. A hierarchical Transformer module serves as the encoder for this model, extracting global context features from images. A convolutional decoder module, consisting of the convolutions and up-sampling layers, is designed to restore spatial dimensions of the feature map, reduce the number of feature channels, and ultimately produce the segmentation results. Furthermore, a U-shaped full-scale feature extraction module is constructed to connect the corresponding layers of the encoder and decoder for capturing fine-grained details and broader contextual information. 

![模型结构-Segformer-Unet-0702](https://github.com/user-attachments/assets/41ce363e-23c5-4969-897a-d7d7ead38da5)
                                                                           
Figure 1. Architecture of the proposed Trans-UNet2 model

The U-shaped full-scale feature extraction (U-FFE) module in the Trans-UNet2 model serves as a bridging module between the Transformer encoder and the convolutional decoder (see Figure 1). Inspired by the architecture of the UNet 3+ network, the U-FFE module is designed with principles of full-scale feature aggregation and hierarchical structuring, as illustrated in Figure 2. This module initially extracts both high-level semantic information and low-level spatial details from the encoder feature layers and then delivers a comprehensive feature representation to the decoder. It is worth noting that the sub-blocks within the U-FFE module are designed with varying depths to accommodate feature maps of different sizes. This design ensures that the model maintains appropriate contextual information and crucial feature details during feature extraction.

![U型链接6](https://github.com/user-attachments/assets/0320f898-a862-4a7a-9cd0-b18657839f86)

Figure 2. Architecture of the proposed U-shaped full-scale feature extraction (U-FFE) module

# The training weights for the Trans-Unet2 model can be accessed via this link:
https://drive.google.com/file/d/1bvKRlljkYc_Tyl52f1Lkn8TTKmY3TWJu/view?usp=sharing

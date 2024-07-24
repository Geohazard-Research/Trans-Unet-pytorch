# Trans-Unet2-pytorch
The overall architecture of the Trans-UNet2 model, as depicted in Figure 1, employs a symmetric encoder-decoder structure similar to that of the UNet model. A hierarchical Transformer module serves as the encoder for this model, extracting global context features from images. A convolutional decoder module, consisting of the convolutions and up-sampling layers, is designed to restore spatial dimensions of the feature map, reduce the number of feature channels, and ultimately produce the segmentation results. Furthermore, a U-shaped full-scale feature extraction module is constructed to connect the corresponding layers of the encoder and decoder for capturing fine-grained details and broader contextual information. 

![image](https://github.com/user-attachments/assets/afad0356-8925-4edb-8468-d057bc81d9b1)
Figure 1. Architecture of the proposed Trans-UNet2 model

The U-shaped full-scale feature extraction (U-FFE) module in the Trans-UNet2 model serves as a bridging module between the Transformer encoder and the convolutional decoder (see Figure 1). Inspired by the architecture of the UNet 3+ network, the U-FFE module is designed with principles of full-scale feature aggregation and hierarchical structuring, as illustrated in Figure 2. This module initially extracts both high-level semantic information and low-level spatial details from the encoder feature layers and then delivers a comprehensive feature representation to the decoder. It is worth noting that the sub-blocks within the U-FFE module are designed with varying depths to accommodate feature maps of different sizes. This design ensures that the model maintains appropriate contextual information and crucial feature details during feature extraction.

![image](https://github.com/user-attachments/assets/b5f7fe96-eb8c-494b-86bb-7d203a6fd80b)
Figure 2. Architecture of the proposed U-shaped full-scale feature extraction (U-FFE) module

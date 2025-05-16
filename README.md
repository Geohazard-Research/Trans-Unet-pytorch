# Trans-UNet Model for Landslide Mapping
The overall architecture of the Trans-UNet model, as depicted in Figure 1, employs a symmetric encoder-decoder structure similar to that of the UNet model.  A hierarchical Transformer module serves as the encoder for this model, extracting global context features from images.  A convolutional decoder module, consisting of convolutions and up-sampling layers, is designed to restore the spatial dimensions of the feature map, reduce the number of feature channels, and ultimately produce the segmentation results.  Furthermore, a U-shaped full-scale feature extraction module (Figure 2) is constructed to connect the corresponding layers of the encoder and decoder for capturing and aggregating fine-grained details information.

![Fig  1](https://github.com/user-attachments/assets/5e21ba90-56c5-47cb-96f2-2466c39d964c)

                                                                     
Figure 1. Architecture of the proposed model


![Fig  2](https://github.com/user-attachments/assets/830b9168-5316-4b7c-9f3b-2c2908ae97d0)

Figure 2. Architecture of the U-shaped full-scale feature extraction (U-FFE) module

# The training weights for the Trans-Unet model can be accessed via this link:
https://drive.google.com/file/d/1bvKRlljkYc_Tyl52f1Lkn8TTKmY3TWJu/view?usp=sharing

# Deep Learning-Based Framework for High-Resolution Mosaic Generation and Efficient Image Compression of Satellite Data
Original README from the Satlas project: [[link]](satlas_README.md)

| ![Super-resolved examples](segmentation/plots/inputs.png) |
|:---------------------------------------------------------:| 
|       *Fig.1: Output examples for compared models.*       |

## Demos
- Super-resolved inputs for Innopolis ROI: [[link]](TODO)
- Super-resolved inputs for Stanford ROI: [[link]](TODO)

## Dataset
|                                  ![RGB Input](figures/new/rgb_input.png)                                  |
|:---------------------------------------------------------------------------------------------------------:| 
| *Fig.2: Example input during training including RGB channels & OSM masks for object-aware discriminator.* |

|                                     ![NIR Input](figures/new/nir_input.png)                                     |
|:---------------------------------------------------------------------------------------------------------------:| 
| *Fig.3: Example input during training including NIR, R, G channels & OSM masks for object-aware discriminator.* |

This project uses a subsample of the latest version of [S2-NAIP dataset](https://huggingface.co/datasets/allenai/s2-naip). 
The prepared subsample is published on Kaggle: [[link]](https://www.kaggle.com/datasets/evgeniyivankin/s2-naip-5k-pairs-rgb-nir).

## Model checkpoints
<!-- TODO: upload to HF/etc -->

## Experiments with segmentation
|                              ![Segmentation examples](segmentation/plots/upp_masks.png)                              |
|:--------------------------------------------------------------------------------------------------------------------:| 
| *Fig.4: Unet++ buildings mask predictions on different inputs including NAIP, Sentinel-2 and super-resolved images.* |
See [`./segmentation`](segmentation) folder for implementation details.

## Experiments with tile seams
<!-- TODO: figures -->
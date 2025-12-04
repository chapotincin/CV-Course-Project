# Galaxy Classification

## Instructions For Use:
To run the code we used Google Colab and its gpus.

First install the required libraries:
!pip install torch torchvision tensorboard numpy

Have your train_evaluate_CNN.py and ConvNet.py and image folder path available. In command line change data_path to point to the folder for which classification(e_s_classifier, roundness_classifier, arms classifier) you want to run and which has the train and test folders. The files inside that folder must be named train and test.  An example is shown below for the command line argument. Change number of epochs, learning rate, batch size as you please. If you wish to use pretrained weights from a saved .pth file (which one is included in our github repository feature_extractor.pth) set to --load_feature_extractor otherwise if you want to save the parameters use --save_feature_extractor

--debug_log prints out what the model guessed vs what it actually was. 

–save_graphs saves the graphs for you.  
!python train_evaluate_CNN.py --data_path "/content/drive/MyDrive/galaxies/type" --num_epochs 30 --learning_rate 0.01 --batch_size 64 --save_graphs --debug_log --load_feature_extractor

## Required Libraries:

`torch`
`torchvision`
`numpy`
`matplotlib`

Install with `pip install torch torchvision numpy matplotlib`.

# Data Sources

Original Dataset: [Zenodo](https://zenodo.org/records/3565489#.Y3vFKS-l0eY)

Objid to Classifications Data: [Galaxy Zoo 2](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz)

## Resulting Image Dataset:

[All Filtered Images](https://drive.google.com/drive/folders/10qCqWFHPoPNe2Tbj1vegkPiwSQGtHRt3)

Elliptical vs. Spiral Classification:
[Train/Test](https://drive.google.com/drive/folders/1WaHt-QjC-Rqpo6_QAh2ItabbJnxzMjMC?usp=sharing)

Roundness Classification:
[Train/Test](https://drive.google.com/drive/folders/1sNikyuK42-hoaMfp8EHQpVRLa5VYU4sD?usp=sharing)

Number of Arms Classification:
[Train/Test](https://drive.google.com/drive/folders/1UP7BRmON2uriyZjDCaop9dYsUQ-1n62H?usp=sharing)

## Sorting the Images

`Sort_Data_into_folder.py` takes the image id and associated classifications from `image_id_data.xlsx` and filters the images from the overall `images_filtered` file into their respective class files; i.e. spiral vs. elliptical.

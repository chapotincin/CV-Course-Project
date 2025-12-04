# Galaxy Classification

## Instructions For Use:
To run the code we used google co lab and its gpus.
First install the required libraries:
!pip install torch torchvision tensorboard numpy
Have your train_evaluate_CNN.py and ConvNet.py and image folder path available. In command line change data_path to point to the folder for which classification you want to run and which has the train and test folders. The files inside that folder must be named train and test.  For example as shown below. Change number of epochs, learning rate, batch size as you please. If you wish to use pretrained weights from a saved .pth file (which one is included in our github repository feature_extractor.pth) set to --load_feature_extractor otherwise if you want to save the parameters you should use --…
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
[Train](https://drive.google.com/drive/folders/1Xp8tzf35JJROHjW2tac0DUVDE16YlYg8) | [Test](https://drive.google.com/drive/folders/167VcOcN99xMUSPLZ3G0ZW5PKK2MTcB48)

Roundness Classification:
[Train](https://drive.google.com/drive/folders/1ruEh-9p4ij70ZFvtXYAf52J6pWi99f4f) | [Test](https://drive.google.com/drive/folders/12mszkEJ6XhyT32Ls67CwmXnA4BiV45Yb)

Number of Arms Classification:
[Train](https://drive.google.com/drive/folders/1iCQb619Q966hA-3PqwB4Mr_NW6-O6Iqk) | [Test](https://drive.google.com/drive/folders/1anHYWCZffnYbfZXMDG0a0nXpK5VE5OM-)

## Sorting the Images

`Sort_Data_into_folder.py` takes the image id and associated classifications from `image_id_data.xlsx` and filters the images from the overall `images_filtered` file into their respective class files; i.e. spiral vs. elliptical.

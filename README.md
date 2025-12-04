Galaxy Classification

Instructions:


Required libraries:
torch
torchvision
numpy
matplotlib

Original Dataset we modified image id to objid
https://zenodo.org/records/3565489#.Y3vFKS-l0eY
objid to classifications data
https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz
Overall Image dataset after cleaning images_filterd
https://drive.google.com/drive/folders/10qCqWFHPoPNe2Tbj1vegkPiwSQGtHRt3?usp=drive_link
Train/Test Ellipicatal vs Spiral Classification
https://drive.google.com/drive/folders/1Xp8tzf35JJROHjW2tac0DUVDE16YlYg8?usp=drive_link
https://drive.google.com/drive/folders/167VcOcN99xMUSPLZ3G0ZW5PKK2MTcB48?usp=drive_link
Train/Test Roundness Classification
https://drive.google.com/drive/folders/1ruEh-9p4ij70ZFvtXYAf52J6pWi99f4f?usp=drive_link
https://drive.google.com/drive/folders/12mszkEJ6XhyT32Ls67CwmXnA4BiV45Yb?usp=drive_link
Train/Test Number of arms Classification
https://drive.google.com/drive/folders/1iCQb619Q966hA-3PqwB4Mr_NW6-O6Iqk?usp=drive_link
https://drive.google.com/drive/folders/1anHYWCZffnYbfZXMDG0a0nXpK5VE5OM-?usp=drive_link

Sort_Data_into_folder.py takes the image id and associated classifications from image_id_data.xlsx and filters the images from the overall images_filtered file into their respective class files. ie (spiral vs elliptical)

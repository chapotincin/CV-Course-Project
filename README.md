# Galaxy Classification

## Instructions For Use:


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

Elliptical vs. Spiral Classification:

[Train](https://drive.google.com/drive/folders/1Xp8tzf35JJROHjW2tac0DUVDE16YlYg8?usp=drive_link) | [Test](https://drive.google.com/drive/folders/167VcOcN99xMUSPLZ3G0ZW5PKK2MTcB48?usp=drive_link)

Roundness Classification:

[Train](https://drive.google.com/drive/folders/1ruEh-9p4ij70ZFvtXYAf52J6pWi99f4f?usp=drive_link) | [Test](https://drive.google.com/drive/folders/12mszkEJ6XhyT32Ls67CwmXnA4BiV45Yb?usp=drive_link)

Number of arms Classification:

[Train](https://drive.google.com/drive/folders/1iCQb619Q966hA-3PqwB4Mr_NW6-O6Iqk?usp=drive_link) | [Test](https://drive.google.com/drive/folders/1anHYWCZffnYbfZXMDG0a0nXpK5VE5OM-?usp=drive_link)

## Sorting the Images

`Sort_Data_into_folder.py` takes the image id and associated classifications from `image_id_data.xlsx` and filters the images from the overall `images_filtered` file into their respective class files; i.e. spiral vs. elliptical.

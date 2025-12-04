# Galaxy Classification
This project trains and tests a neural network to classify galaxy shape, structure, and number of arms. It allows training and using a shared feature across all three tasks.

## Instructions For Use
We used Google Colab and its GPUs to run this code.

First, install the required libraries as detailed in "Required Libraries" below.

1. Place `train_evaluate_CNN.py` and `ConvNet.py` in the directory in which you wish to run the program. 

2. Ensure that your current environment has access to your desired train and test data folders. Use the `--data_path` command line option to point to the folder containing the `train` and `test` data folders, named as such, with the data inside that divided into individual class folders.

3. Set the number of epochs, learning rate, and batch size using the command line options as in the example below.

4. If you wish to use pretrained weights from a saved `.pth` file (an example one, trained for 60 epochs on the galaxy type data, is included as `feature_extractor.pth`), add the flag `--load_feature_extractor`. If you want to save the parameters, use `--save_feature_extractor`. Use the `--feature_extractor_path` option to set the path to read the feature extraction from or write it to. 

Example: `python train_evaluate_CNN.py --data_path "/path/to/root/folder/of/data" --num_epochs 30 --learning_rate 0.01 --batch_size 64 --save_graphs --debug_log --load_feature_extractor --feature_extractor_path "/path/to/feature/extractor"`

## Required Libraries

`torch`
`torchvision`
`numpy`
`matplotlib`

Install with `pip install torch torchvision numpy matplotlib`.

## Example Data Layout

```
.../
  test/
    class1/
      a.jpg
      b.jpg
      ...
    class2/
      c.jpg
      d.jpg
      ...
  train/
    class1/
      e.jpg
      f.jpg
      ...
    class2/
      g.jpg
      h.jpg
      ...
```

## Command Line Options

 `--learning_rate`:
 
   type: float, 
   
   default: 0.001,
   
   description: Initial learning rate.
   
 `--num_epochs`:
 
   type: int,
   
   default: 60,
   
   description: Number of epochs to run trainer. 
   
 `--batch_size`:
 
   type: int, 
   
   default: 10,
   
   description: Batch size. Must divide evenly into the dataset sizes. 
   
 `--debug_log`:
 
   Set to `True` if present. 
   
   default: False,
   
   description: Increase logging for debugging. 
   
 `--save_graphs`:
 
   Set to `True` if present. 
   
   default: False,
   
   description: Saves output graphs to a file in the current working directory. 
   
 `--save_feature_extractor`:
 
   Set to `True` if present. 
   
   default: False,
   
   description: Saves the feature extractor model to a file in the current working directory. 
   
 `--save_model`:
 
   Set to `True` if present. 
   
   default: False,
   
   description: Saves the full model to a file in the current working directory. 
   
 `--load_feature_extractor`:
 
   Set to `True` if present. 
   
   default: False,
   
   description: Loads the feature extractor model from a file in the current working directory. 
   
 `--load_model`:
 
   Set to `True` if present. 
   
   default: False,
   
   description: Loads the full model from a file in the current working directory. 
   
 `--data_path`:
 
   type: str,
   
   default: `/content/drive/MyDrive/galaxies/type`:
   
   description: Directory containing the /train and /test data folders. 
   
 `--feature_extractor_path`:
   type: str,
   
   default: "feature_extractor.pth",
   
   description: The path to the feature extractor's weights, for saving or loading. 
   
 `--model_path`:
 
   type: str,
   
   default: "model.pth",
   
   description: The path to the model's weights, for saving or loading. 
   

# Data Sources

Original Dataset: [Zenodo](https://zenodo.org/records/3565489#.Y3vFKS-l0eY)

Objid to Classifications Data: [Galaxy Zoo 2](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz)

## Resulting Image Dataset

[All Filtered Images](https://drive.google.com/drive/folders/10qCqWFHPoPNe2Tbj1vegkPiwSQGtHRt3)

Elliptical vs. Spiral Classification:
[Train/Test](https://drive.google.com/drive/folders/1WaHt-QjC-Rqpo6_QAh2ItabbJnxzMjMC?usp=sharing)

Roundness Classification:
[Train/Test](https://drive.google.com/drive/folders/1sNikyuK42-hoaMfp8EHQpVRLa5VYU4sD?usp=sharing)

Number of Arms Classification:
[Train/Test](https://drive.google.com/drive/folders/1UP7BRmON2uriyZjDCaop9dYsUQ-1n62H?usp=sharing)

## Sorting the Images

`Sort_Data_into_folder.py` takes the image id and associated classifications from `image_id_data.xlsx` and filters the images from the overall `images_filtered` file into their respective class files; i.e. spiral vs. elliptical.

# Sperm-Segmentation-Project

## Quick Tour:<br>
This is a pytorch implementation for our MICCAI-2024 paper "CS3: Cascade SAM for Sperm Segmentation".<br>
To run the code, please make sure you have prepared your data following the same structure as follows (you can also refer to the examplar data in this repository).<br>

## Project Structure<br>
Here is an overview of the project's directory structure:<br>
```
Sperm-Segmentation-Project/
    - main.py            # Main script for model training
    - image_processing.py            # Some functions
    - mask_processing.py            # Some functions
    - skeleton_analysis.py            # Some functions
    - utils.py            # Some functions
    - segment-anything-main/       #SAM model
        - sam_vit_h_4b8939.pth
    - README.md             # This file
    - requirements.txt      # The required Python packages
    - data/                 # Directory for storing your dataset
        - original_images     # Original images
        - Preprocessing_images      # Preprocessing images

```
## Methodology:<br>

<div align="center">
    <img src="Example Result/method2.png" alt="drawing" width="600"/>
</div>

## Data Organization:<br>
Our data contains about 4 sperm pictures, and the resolution of each picture is 720*540. <br>
Each image contains several intact sperms, and they often overlap.<br>
To structure the data for training and testing, it is recommended to organize dataset as follows:<br>
```
- data/                 # Directory for storing your dataset
        - original_images     # Original images
            - 000.jpg
            - 001.jpg
            - ...
            - N.jpg
        - Preprocessing_images      # Preprocessed images
```
Here's how the data is organized:<br>

- 'original_images' contains original images you want to process.
- 'Preprocessing_images' contains preprocessing images.<br>



We provide several samples for testing.  The resolution of the image specification is 720*540.<br>


## Getting Started<br>
Follow these steps to set up and start working with the project:<br>

1. **Environment Setup**: Create a virtual environment and install the required dependencies. Step omitted here.<br>
2. **SAM model Setup**:The code requires SAM model and its weight file. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended. Download SAM model and the weight file from [this place](https://github.com/facebookresearch/segment-anything). After downloading and unpacking:<br>

Install Segment Anything:

```
cd segment-anything-main; pip install -e .
```

  The following optional dependencies are necessary for mask post-processing, saving masks in COCO format, the example notebooks, and exporting the model in ONNX format. `jupyter` is also required to run the example notebooks.

```
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```
Please download SAM model and the weight file to the path in the folder segment-anything-main. If it is not achieved，please modify the paths following.<br>
4. **Check the path of sam**:Replace the paths of the SAM model and weight file in the code with the path you actually installed to.<br>
<div align="center">
    <img src="Example Result/code.png" alt="drawing" width="500"/>
</div>

5. **Run the code**： Enter the directory ：<br>
```
cd Sperm-Segmentation-Project-main
```
Run the program by command :<br>
```
python Sperm.py
```

## Example Result:<br>
After running the code, we will obtain the json file of the segmentation result in the folder ```-original_images``` . Here are examples of what our visualization looks like.<br>
<div align="center">
    <img src="Example Result/1191.jpg" alt="drawing" width="300"/>
    <img src="Example Result/0850.jpg" alt="drawing" width="300"/>
</div>


## References：<br>
- [segment-anything](https://github.com/facebookresearch/segment-anything) [1]
















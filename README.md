# Document Image Processing Pipeline

This project implements a computer vision pipeline for processing document images. It detects document boundaries in various lighting conditions, applies perspective transformation to straighten documents, and enhances image quality with multiple post-processing techniques.

## Project Structure

- **`/src`**: Source code directory
  - `main.py`: Main script for document detection and mask generation
  - `postProcessing.py`: Script for perspective transformation and image enhancement
- **`/outputs`**: Generated document masks
- **`/results`**: Final processed images
- **`/datasets`**: Original document images

## Features

1. **Document Detection**

   - Adaptive contour detection to identify document boundaries
   - Works in various lighting conditions (with/without flash, with/without ambient light)
   - Handles blurry images and different document types

2. **Perspective Transformation**

   - Converts angled document views to straight, properly oriented documents
   - Maintains proper document proportions
   - Removes unnecessary background

3. **Image Enhancement**
   - White balance correction
   - Sharpening and detail enhancement
   - Noise reduction
   - Edge enhancement with Sobel operators
   - Morphological operations for text clarity
   - Top-Hat transformation for improved contrast

## How to Use

### Basic Processing

To process all images in the dataset:

```bash
python src/main.py
```

This will:

1. Detect document boundaries in all images
2. Generate mask files in `/outputs`

Then, to apply perspective transformation and enhance images:

```bash
python src/postProcessing.py
```

This will:

1. Apply perspective transformation based on the detected masks
2. Enhance image quality
3. Save results in the `/results` directory

## Requirements

- Python 3.11+
- OpenCV
- NumPy
- Pandas
- matplotlib (for visualization)
- rich (for console output)
- Jupyter (for interactive notebook) s

## Installation

```bash
pip install opencv-python numpy pandas matplotlib rich jupyter
```

## Results

The pipeline produces:

- Mask images showing detected document boundaries
- Perspective-corrected document images
- Enhanced document images with improved contrast, sharpness, and readability

# Face Landmarker Fine-tuning

This folder provides utilities for fine-tuning the pre-trained **MediaPipe Face Landmarker** model with custom data. The sample dataset in this repository assumes each 3D scan provides 16 facial landmark points (indexed from 1 to 16) stored in OBJ files.

## Requirements

- Python 3.9+
- [MediaPipe Model Maker](https://developers.google.com/mediapipe/solutions/model_maker)
- TensorFlow 2.x
- [trimesh](https://trimsh.org/) for converting OBJ files

Install dependencies with pip:

```bash
pip install mediapipe-model-maker tensorflow trimesh
```

## Preparing the Dataset

1. Put all your OBJ files in a directory. Each file should contain the facial scan with the first 16 vertices representing the landmark points.
2. Run the dataset preparation script to render each scan to an image and generate a corresponding JSON annotation file:

```bash
python3 prepare_dataset.py \
  --input_dir path/to/obj_files \
  --output_dir path/to/output_dataset
```

The output directory will contain PNG images and JSON files with landmark coordinates. These files follow the structure expected by `mediapipe_model_maker.face_landmarker.Dataset.from_folder`.

## Training

Run the training script with the prepared dataset:

```bash
python3 train.py \
  --model_path face_landmarker.task \
  --train_dir path/to/output_dataset \
  --validation_dir path/to/output_dataset \
  --output_dir output_model
```

The fine-tuned model will be saved in the specified `output_dir`.

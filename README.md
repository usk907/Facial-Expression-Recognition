# Facial Expression Recognition using Machine Learning

This repository contains the code and dataset for training and deploying a machine learning model to detect facial expressions. The model is capable of recognizing five different facial expressions: angry, fear, happy, sad and surprise.

## Model

The facial expression detection model is based on the EfficientNetB0 architecture, a convolutional neural network (CNN) known for its efficiency and effectiveness in image classification tasks.

## Dataset

The dataset used for training the facial expression detection model is included under the `dataset/` directory. It consists of two subdirectories: `train/` for training images and `test/` for testing images. Each subdirectory contains images categorized into folders based on their respective facial expressions.

## Files

- **Model_Training.py**: Python script for training the facial expression detection model. It uses the EfficientNetB0 architecture and TensorFlow/Keras for model training.
- **Photo_Input.py**: Python script for making predictions on a single input image using the trained model.
- **Live_Video_Input.py**: Python script for performing real-time facial expression detection on live video input from a webcam.
- **Model.h5**: Trained model saved in HDF5 format.

## Usage

1. **Training the Model**: Execute `Model_Training.py` to train the facial expression detection model using the provided dataset.
2. **Making Predictions on Images**: Use `Photo_Input.py` to make predictions on a single input image by specifying its file path.
3. **Real-time Video Input**: Run `Live_Video_Input.py` to perform real-time facial expression detection on live video input from a webcam.

## Dependencies

Install the following Python packages using pip:

```bash
pip install numpy pandas opencv-python tensorflow
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

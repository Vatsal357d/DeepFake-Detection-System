# DeepFake-Detection-System

Deepfake Detection Model
This repository contains a Python implementation for training a deep learning model to detect deepfake images. The model is built using TensorFlow and Keras and can classify images as either "Real" or "Fake."

Dataset
The dataset should be organized into the following directory structure:

deepfake_dataset/
├── train/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
train/ contains training images in real/ and fake/ subdirectories.
test/ contains testing images in real/ and fake/ subdirectories.
Ensure the dataset is preprocessed and organized correctly.

Requirements
Install the required Python libraries before running the code:

bash
Copy
Edit
pip install tensorflow opencv-python numpy matplotlib scikit-learn
Code Overview
Data Augmentation
The code uses ImageDataGenerator to apply real-time data augmentation, including:

Rescaling pixel values to [0, 1].
Random shear and zoom transformations.
Random horizontal flips.
Model Architecture
The Convolutional Neural Network (CNN) consists of:

Three convolutional layers with ReLU activation and max pooling.
A fully connected dense layer with 128 neurons and a dropout layer to reduce overfitting.
A final dense layer with a sigmoid activation for binary classification.
Training and Evaluation
The model is trained using the Adam optimizer and binary cross-entropy loss.
Training metrics include accuracy and validation accuracy.
After training, the model is evaluated on the test dataset.
Model Saving
The trained model is saved to deepfake_detector_model.h5.

Usage
Training the Model
Place your dataset in the specified directory structure.
Run the script to train the model:
bash
Copy
Edit
python train.py
Model Evaluation
The script outputs:

Test accuracy.
A classification report with precision, recall, and F1-score.
A confusion matrix for "Real" and "Fake" predictions.
Visualization
The script generates a plot showing training and validation accuracy over epochs.

Example Outputs
Test Accuracy: XX.XX%
Confusion Matrix:
lua

[[TP FP]
 [FN TN]]
Predictions
The model predicts whether images in the test dataset are "Real" or "Fake."

Results
Accuracy: Achieved during model evaluation.
Model File: deepfake_detector_model.h5.

Contact
For questions or suggestions, please open an issue or contact the repository owner.

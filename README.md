# Project Overview: Building a Model to Identify My Face and Other People's Faces

## Purpose
The goal of this project is to develop a deep learning model using computer vision techniques to distinguish between my own face and faces of other individuals. Specifically, a convolutional neural network (CNN) will be employed for image classification.

## Model Details
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Evaluation Metric**: Accuracy
- **Dropout Rate**: 0.5

## Model Architecture
- **Convolutional Layers (Conv2D)**:
  - Filters: 32, 64, 128, 128
  - Kernel Size: (3, 3)
  - Activation Function: ReLU
- **Pooling Layers (MaxPooling2D)**:
  - Pool Size: (2, 2)
- **Flatten Layer (Flatten)**: Converts the output of convolutional layers into a 1D array
- **Fully Connected Layers (Dense)**:
  - Units: 512
  - Activation Function: ReLU
- **Dropout Layer (Dropout)**: Dropout rate of 0.5
- **Output Layer (Dense)**:
  - Units: 1
  - Activation Function: Sigmoid

### build_image_classification_cnn.py

This script loads images from Google Drive, detects faces using Haar Cascade classifier, crops and resizes detected faces, and saves them.

#### Key Operations

- **Mount Google Drive**: Mounts Google Drive to access image data.
- **Set Image Folders**: Specifies folders for processing images and saving processed images.
- **Face Detection**: Uses Haar Cascade classifier to detect faces in images.
- **Face Cropping and Resizing**: Crops detected faces with padding and resizes them to 150x150 pixels.
- **Save Resized Face Images**: Saves cropped and resized face images to specified folder.

---

### evaluate_confusion_matrix.py

This script evaluates the test dataset using a pre-trained model, generates a confusion matrix, and computes and displays performance metrics of the model.

#### Key Operations

- **Mount Google Drive**: Mounts Google Drive to load model and data.
- **Load Model**: Loads pre-trained model.
- **Prepare Test Data**: Normalizes test data and sets up data generator for evaluation.
- **Evaluate Model**: Evaluates model on test data and displays accuracy.
- **Get Predictions**: Retrieves predictions from the model and compares with actual labels.
- **Compute Confusion Matrix and Metrics**: Computes and displays confusion matrix, accuracy, precision, recall, and F1 score.
- **Plot Confusion Matrix**: Visualizes confusion matrix for better understanding.

---

### output_misclassified_images.py

This script displays misclassified images, showing actual and predicted labels for each image.

#### Key Operations

- **Get Misclassification Indices**: Retrieves indices of images misclassified by the model.
- **Define Misclassified Image Display Function**: Defines a function to display misclassified images and show actual and predicted labels for each image.
- **Process Image Display**: Retrieves paths and names of misclassified images, loads and converts images to RGB format, displays each image as a subplot with titles showing actual and predicted labels.
- **Function Execution**: Calls the function to display all misclassified images.

---

### face_region_resizer.py

This script detects faces in images within a specified directory, resizes each detected face, and saves them.

#### Key Operations

- **Mount Google Drive**: Mounts Google Drive to access image data.
- **Set Folder Paths**: Specifies input and output folder paths.
- **Load Haar Cascade Classifier**: Loads Haar Cascade classifier for face detection.
- **Process Images**: For each image in the folder, detects faces, pads around the faces, crops them, resizes them to the specified size (150x150 pixels), and saves them. Also displays the cropped face images for verification.

---

### image_augmentor.py

This script applies brightness adjustment, saturation adjustment, and blur to generate new images for given image files within a specified directory.

#### Key Operations

- **Mount Google Drive**: Mounts Google Drive to access image data.
- **Set Directory Paths**: Specifies input directory and output directories for processed images with adjusted brightness, saturation, and blur.
- **Set Image Processing Parameters**: Configures parameters for brightness adjustment, saturation adjustment, and blur.
- **Process Images**: For each image in the directory, adjusts brightness, adjusts saturation, applies blur, generates new images, and saves them in the specified output directories.

# Mini Food Project (Food Vision Big)

## Description

The Mini Food Project (also referred to as "Food Vision Big") is a Python-based image classification project that recognizes 101 distinct food categories using the Google Food-101 dataset. Leveraging TensorFlow and TensorFlow Datasets, this project demonstrates end-to-end machine learning workflows including data loading, visualization, preprocessing, model training, and evaluation, with experiment tracking via Weights & Biases (W\&B).

## Table of Contents

* [Features](#features)
* [Tech Stack](#tech-stack)
* [Dataset](#dataset)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Results](#results)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Features

* Load and preprocess the Food-101 dataset using TensorFlow Datasets
* Batch, shuffle, and normalize images for efficient training
* Data visualization: sample images, loss & accuracy curves
* Build and train convolutional neural networks (CNNs) in TensorFlow
* Track experiments and metrics with Weights & Biases (W\&B)
* Compute and plot confusion matrices for performance analysis

## Tech Stack

* **Python 3.8+**
* **TensorFlow** (including `tensorflow_datasets`)
* **NumPy**
* **Matplotlib**
* **scikit-learn** (for confusion matrix)
* **Weights & Biases (W\&B)**

## Dataset

This project uses the [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) which contains 101 food categories with 1,000 images each. The dataset is automatically downloaded and prepared via TensorFlow Datasets.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/mini-food-project.git
   cd mini-food-project
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate       # On Windows: venv\\Scripts\\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Log in to Weights & Biases to enable experiment tracking:

   ```bash
   wandb login
   ```

## Usage

The main workflow is implemented in a Jupyter notebook (`Mini_Food_Project.ipynb`). To run it:

1. Launch Jupyter:

   ```bash
   jupyter notebook
   ```

2. Open `Mini_Food_Project.ipynb` and execute the cells in order.

Key sections in the notebook:

* **Data Loading & Inspection**: Load the Food-101 dataset and visualize samples.
* **Preprocessing**: Define functions to batch, shuffle, and normalize data.
* **Model Definition**: Create CNN architectures using `tf.keras`.
* **Training**: Train the model with callbacks including `WandbCallback`.
* **Evaluation**: Plot loss/accuracy, and generate confusion matrices.

## Project Structure

```
mini-food-project/
│
├── Mini_Food_Project.ipynb  # Jupyter notebook containing the full workflow
├── helper_functions.py      # Custom plotting functions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation (this file)
```

## Results

The model achieves competitive accuracy on the Food-101 dataset. Sample plots of training/validation loss and accuracy, as well as the confusion matrix, are generated in the notebook. You can find interactive visualizations on your W\&B dashboard if experiment tracking is enabled.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

* **Food-101 Dataset** by Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.
* **TensorFlow Datasets** team for easy dataset integration.
* **Weights & Biases** for experiment tracking.

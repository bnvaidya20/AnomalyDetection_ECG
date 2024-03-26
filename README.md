# Anomaly Detection on ECGs

## Overview

This project focuses on anomaly detection in electrocardiograms (ECGs) using an autoencoder neural network. The objective is to identify abnormal heartbeats from ECG data, which can be crucial for early detection of heart diseases.

## Datasets

The **ECG5000** dataset, containing 5,000 ECGs with 140 data points each, is used in this project. It includes specific labels for normal and abnormal rhythms, making it ideal for supervised anomaly detection tasks.

The dataset is available [here](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).

### Original Dataset Source

Originating from a 20-hour long ECG record from a patient with severe congestive heart failure (from the BIDMC Congestive Heart Failure Database, record "chf07"), the dataset was processed to extract individual heartbeats and normalize their lengths.

For more details, refer to the following publications:
- Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals." Circulation 101(23).
- "A general framework for never-ending learning from time series streams", DAMI 29(6).

## Project Structure

- `data/`: Contains the dataset file.
- `src/`: Houses the source code for model training and evaluation.
- `README.md`: Provides a project overview and setup instructions.

## Getting Started

Ensure the installation of the following dependencies:
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn


## Codes

This project's codebase is inspired by and adapted from the following Kaggle notebooks:

- [ECG Anomaly Detection](https://www.kaggle.com/code/mineshjethva/ecg-anomaly-detection/notebook)
- [Anomaly Detection using AutoEncoder on ECG5000](https://www.kaggle.com/code/itzsanju/anomaly-detection-using-autoencoder-on-ecg5000)

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

## Acknowledgments

Special thanks to the authors of the referenced Kaggle notebooks and the contributors to the Physionet database for making the ECG5000 dataset accessible for academic and research purposes.

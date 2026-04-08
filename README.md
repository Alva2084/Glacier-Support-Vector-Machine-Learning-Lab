# Glacier-Support-Vector-Machine-Learning-Lab
The purpose of this project is to implement Support Vector Machine Learning (SVM) to classify GIS data into two categories: glacier and non-glacier areas.

## Features
- Programming Language: Python
- Data Loading and Preprocessing Pipeline
- Automatic Removal of Unrelevant Data
- Adding Missing Values via Average Values
- Standardization for SVM
- Stratified Sampling
- Linear SVM
- RBF SVM
- Evaluation Metrics

## Project Structure
- glacierSVM.py
- main.py
- test_GlacierSVM.py
- glacier_land_ice102.csv
- README.md

## Requirements
- Python 3.10 or higher
- PyCharm or Terminal
- pip
  - Install required Packages:
    - pip install pandas scikit-learn pytest numpy

## How to Run
To run the test:
- In the terminal (in terminal for PyCharm as well) bash: pytest -v test_GlacierSVM.py

To Run the Code:
- In the terminal:
  - bash python main.py
- In PyCharm
  - Run the main.py


# Electricity Bill Prediction

This project aims to predict the electricity bill of a building based on various features such as the number of fans, lights, ACs, the city the building is in, and the company of electric appliances used.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Accurate prediction of electricity bills is essential for budget planning and energy management. This project uses machine learning techniques to predict the electricity bill of a building. The model is trained using features that significantly influence electricity consumption.

## Features Used

The dataset includes the following features:

- **Number of Fans**: The total number of fans in the building.
- **Number of Lights**: The total number of lights in the building.
- **Number of ACs**: The total number of air conditioners in the building.
- **Refrigirator**: The total number of Refrigirators in the building.
- **TV**: The total number of air TV's in the building.
- **Monitor**: The total number of Monitors in the building.
- **Montly Hours**:Number of Hours the building uses electricity each month. 
- **Tarrif Rate**:Price charged by an electricity provider for the consumption of electricity.
- **Months**: Number of months the building used Electricity.
- **City**: The city where the building is located.
- **Company of Electric Appliances**: The brand or company of the electric appliances used in the building.

## Installation

To run this project, you need to have Python installed on your system. You can install the required libraries using `pip`.

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/svnsaisathvik/Predicting-the-Electricity-Bill
    cd Predicting-the-Electricity-Bill
    ```

2. Prepare your dataset and save it as `data.csv` in the project directory.

3. Run the training script:
    ```bash
    python train.py
    ```

4. Make predictions:
    ```bash
    python predict.py --input sample_input.csv --output predictions.csv
    ```

## Model

The model uses a Linear Regression algorithm to predict the electricity bill. The libraries used for this project include:

- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib**: For data visualization.
- **Scikit-Learn**: For machine learning algorithms.

### Training the Model

The model is trained on a dataset with the aforementioned features. The training script performs the following steps:

1. Load the dataset.
2. Preprocess the data (handling missing values, encoding categorical variables, etc.).
3. Split the dataset into training and testing sets.
4. Train the Linear Regression model on the training set.
5. Evaluate the model on the testing set.

## Results

The model's performance is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)


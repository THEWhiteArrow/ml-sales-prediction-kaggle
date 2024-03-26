# Machine Learning Project: Store Sales Prediction

This is a machine learning project focused on predicting sales for the thousands of product families sold at Favorita stores located in Ecuador. The training data includes dates, store and product information, whether the item was being promoted, as well as the sales numbers. Additional files include supplementary information that may be useful in building your models.

## Getting Started

To run this project, ensure you have Python installed on your machine. Install the required dependencies by executing `pip install -r requirements.txt`.
Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and extract it into the `data` directory. Please note that due to the rules of the challenge/competition, the data files should not be shared with individuals who have not agreed to the competition rules.

Navigate to the project's main directory and run the main file using the command `python main.py`.

## Model

The output of the project is a model that predicts the sales of a product in a store for a given date. The model is built using a time series forecasting approach.
It generates predictions for the test data and saves them in a CSV file in the `output` directory under the name `submission_{accuracy}.csv`.

## Dataset

The dataset used for this project includes historical sales data for different stores and products. It can be downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and should be extracted and placed in the `data` directory.

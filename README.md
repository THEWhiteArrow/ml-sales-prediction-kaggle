# Machine Learning Project: Store Sales Prediction

This is a machine learning project focused on predicting sales for the thousands of product families sold at Favorita stores located in Ecuador. The training data includes dates, store and product information, whether the item was being promoted, as well as the sales numbers. Additional files include supplementary information that may be useful in building your models.

## Getting Started

To run this project, ensure you have Python installed on your machine. Install the required dependencies by executing `pip install -r requirements.txt`.
Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and extract it into the `data` directory. Please note that due to the rules of the challenge/competition, the data files should not be shared with individuals who have not agreed to the competition rules.

### Model

The output of the project is a model that predicts the sales of a product in a store for a given date. The model is built using a time series forecasting approach.
It generates predictions for the test data and saves them in a CSV file in the `output` directory.

The project implements a few models that were build to experiment with different approaches to the problem. The models are:

-   Deterministic Process Model (Multiple Problem Apporach):

    -   This model uses a deterministic process to predict the sales of all stores and products families at once. It uses a simple linear regression model to predict the sales of each product family at each store.
    -   Please find it in `optional_submission.py` file in `utils` directory in `solve_optional_submission` function.

-   Deterministic Process Model (Single Problem Approach):

    -   This model uses a deterministic process to predict the sales of a single product family at a single store. It uses a simple linear regression model to predict the sales of a single product family at a single store. To solve the multiple problems it uses a for loop to iterate over all the product families and stores. It is slower than the multiple problem approach but easier to understand and debug.
    -   Please find it in `optional_submission.py` file in `utils` directory in `solve_optional_customized_submission` function.

-   ML Forecasting Model:

    -   This model has multiple target values and uses Direct approach to predict the sales of a single product family at a single store. It uses a Hybrid model that combines a linear regression model with a XGBoost model to predict the sales of a single product family at a single store. To solve the multiple problems it uses a for loop to iterate over all the product families and stores. It is slower than the multiple problem approach but easier to understand and debug.
    -   Please find it in `ml_forecasting.py` file in `utils` directory in `solve_using_ml_forecasting` function.

-   Advance ML Forecasting Model:

    -   This model has multiple target values and uses Direct approach to predict the sales of all stores and all families at once. Features are the most complex in comparison to the other models. The model is to use information from the following files:
        -   `train.csv`
        -   `test.csv`
        -   `oil.csv`
        -   `holidays_events.csv`

    At the current state, the model is still under development. Please check the progress in `advanced.py` file in `root` directory - I might have forgotten to update README and it may be already functional.

### Caching

The project uses a caching mechanism to load the data files into dataframes and store them in the `cache` directory. This is done to speed up the data loading process. If you want to reload the data from the original files, you can delete the files in the `cache` directory.

## Dataset

The dataset used for this project includes historical sales data for different stores and products. It can be downloaded from the [Kaggle competition page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) and should be extracted and placed in the `data` directory.

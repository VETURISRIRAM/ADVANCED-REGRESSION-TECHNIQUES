## House Prices Predictions - Advanced Regression Techniques


#### Description

Machine Learning to predict the house prices. (Regression Problem)

Dataset fetched from [Kaggle's House Prices Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/)


#### How to run the code locally?

Make sure your system has the libraries mantioned in `requirements.txt` file.

The entire code is present in the file `main.py`. You can run the following command to run the file.

```
python main_file.py --data_dir house-prices-advanced-regression-techniques
```


## Major Steps Involved


### Data Preprocessing and Cleaning

#### `Missing Values`: 

The missing values in the numerical and categorical features were replaced by the median and most frequent values of the features respectively.

#### `Categorical Features`: 

Mapped the categorical entries to numbers uniquely. For example, 'male' becomes 0 and 'female' becomes 1.

#### `Standardization`: 

The entire data was normalized.


### Feature Engineering

The correlation coefficient values are calculated and only the top 30 feaures are considered for modeling. 

The threshold was decided experimenatlly.


### Modeling

`XGBoost` was used in this project. It is a very powerful tree boosting algorithm based on the gradients. 

It exploits every chunk of memory available in return, in a very less time, more work is done. 

Also, ensembling helped get better predictions on `test.csv`


### Evaluation Metrics

#### `Root Mean Squared Error`
#### `Mean Absolute Error`
#### `Mean Absolute Percentage Error`
#### `Mean Percentage Error`


### Results

Predictions are stored in `final_submission.csv` with a kaggle score of `0.17408`

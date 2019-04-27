"""
@author: Sriram Veturi
@title: House Prices - Advanced Regression Techniques
@date: 04/26/2019
"""

import os
import operator
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Parse Arguments
parser = argparse.ArgumentParser(description="XGBoost Modeling for House Prices Predictions.")
parser.add_argument(
    "--data_dir", 
    type=str,
    help="Path to the directory in which the data files are stored."
)
args = parser.parse_args()

# Globals..
CLASS_VARIABLE = "SalePrice"
PREDICTIONS_FILE = "final_submission.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SAMPLE_SUBMISSION_FILE = "sample_submission.csv"

global final_column_name


def get_data(file_path):
    """
    Function to get the data files.
    :param file_path: Path where the data files are stored.
    :return train_df: Train set.
    :return test_df: Test set.
    :return sample_submission_df: Submission file.
    """

    train_df = pd.read_csv(os.path.join(file_path, TRAIN_FILE))
    test_df = pd.read_csv(os.path.join(file_path, TEST_FILE))
    sample_submission_df = pd.read_csv(os.path.join(file_path, SAMPLE_SUBMISSION_FILE))
    return train_df, test_df, sample_submission_df


def get_train_info(train_df):
    """
    Function to get some information about the train set.
    :param train_df: Train set.
    :return: None (Just Printing)
    """

    print("Train Dataset Information:")
    print(train_df.info())
    print("\nTrain Dataset Description:")
    print(train_df.describe())


def get_missing_percentages(df):
    """
    Function to get the columns and the missing values percentages.
    :param df: Dataframe.
    :return: Missing Values Percentages Dataframe.
    """

    # Missing Values Analysis
    # Source: https://stackoverflow.com/questions/51070985/find-out-the-percentage-of-missing-values-in-each-column-in-the-given-dataset
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({'column_name': df.columns,
                                     'percent_missing': percent_missing})
    missing_value_df.sort_values('percent_missing', inplace=True)
    print("\nMissing Values Percentages:")
    print(missing_value_df.loc[missing_value_df['percent_missing'] != float(0)])
    return missing_value_df


def fill_mode_median_values(df):
    """
    Function to fill missing values with median and most frequent values.
    :param df: Dataframe.
    :return df: Filled Dataframe.
    """

    # Since the missing values are represented by "NA", I replaced it with Numpy NaN.
    # Replace missing values with numpy nan.
    def replace_na_with_npnan(entry):

        if entry == "NA":

            return np.nan

        else:

            return entry

    # Doing the above operation for all the columns.
    columns = df.columns
    for column in columns:

        df[column] = df[column].map(lambda x: replace_na_with_npnan(x))

    # Handle numeric columns in the dataframe.
    # For missing values, I filled them with the median value in the column.
    numeric_columns = df.select_dtypes(include=["int64", "float64"])
    numeric_df = df[list(numeric_columns)]
    df = df.drop(columns=numeric_columns)
    numeric_df.fillna(numeric_df.median().iloc[0], inplace=True)
    df = pd.concat([df, numeric_df], axis=1)

    # Handle categorical columns in the dataframe.
    # For missing values, I replaced them with the most frequent categorical entry.
    # For this operation, I dropped those columns and considered it a separate dataframe.
    object_columns = df.select_dtypes(include=['object'])
    object_df = df[list(object_columns)]
    df = df.drop(columns=object_columns)
    for column in object_columns:

        entries = list(object_df[column])
        entries = dict(Counter(entries))
        sorted_entries = sorted(entries.items(), key=operator.itemgetter(1))

        most_frequent_entry = sorted_entries[-1][0]

        # This check is for float version of Numpy NaN.
        # The function get_most_frequent() (written below) does not catch the float Numpy NaN.
        # So, whenever there is Float Numpy NaN in the last place, it takes the second last value.
        if type(most_frequent_entry) is float:

            most_frequent_entry = sorted_entries[-2][0]

        # Return most frequent entry if a missing value is found.
        def get_most_frequent(entry):

            if entry == np.nan or entry == "nan":

                return most_frequent_entry

            else:

                return entry

        object_df[column] = object_df[column].map(lambda x: get_most_frequent(x))

    # Concatenate the two tables now as a final table.
    df = pd.concat([df, object_df], axis=1)
    return df


def handle_missing_values(df):
    """
    Function to handle missing values in the dataframe.
    :param df: Dataset with missing values.
    :return df: Dataset with filled missing values.
    """

    missing_value_df = get_missing_percentages(df)
    missing_threshold = 80

    # Only the features with more than 80% of missing values in them. 
    missing_mostly = missing_value_df.loc[missing_value_df['percent_missing'] >= float(missing_threshold)]

    # Drop the columns with more than 80% of missing values
    df = df.drop(columns=missing_mostly['column_name'].values)

    # Fill missing values with mode and median values
    df = fill_mode_median_values(df)
    return df


def handle_categorical_data(df):
    """
    This function encodes the categorcial features of the dataset.
    :param df: DataFrame
    :return: df: Encoded DataFrame
    """

    # Preprocess categorical columns
    # Drop the categorical features and consider the dropped featutes
    # as a separated dataframe and do the operations.
    catData = df.select_dtypes(include=['object'])
    catColumns = catData.columns
    df = df.drop(columns=catColumns)

    # Mapping them to unique numeric values.
    for x in catData.columns:

        uniqueValues = catData[x].unique()

        # Mapping hashmap
        mapping = dict(zip(uniqueValues, np.arange(float(len(uniqueValues)))))
        catData[x] = catData[x].map(mapping)

    df = pd.concat([df, catData], axis=1)
    return df


def standardize_dataset(df):
    """
    This function should standardize/normalize the numerical values.
    :param df: DataFrame
    :return: df: Standardized DataFrame
    """

    # No need to scale the target / class label
    # Thus, drop it, scale the features and then, concatenate the class label again.
    class_variable = df[CLASS_VARIABLE]
    df = df.drop(columns=[CLASS_VARIABLE])
    df_columns = df.columns
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(df)
    df = pd.DataFrame(data=scaledData, columns=df_columns)
    df[CLASS_VARIABLE] = class_variable

    return df


def split_features_and_target(df):
    """
    Function to split the features and class varibles.
    :param df: Dataframe.
    :return features: Features.
    :return target: Class Variable.
    """

    target = df[CLASS_VARIABLE]
    df = df.drop(columns=[CLASS_VARIABLE])
    return df, target


def preprocess_dataframe(df):
    """
    Function to preprocess the dataframe.
    :param dataframe: Dataset to be preprocessed.
    :return dataframe: Preprocessed Dataframe.
    """

    # Handle missing values
    df = handle_missing_values(df)

    # Handle categorical features
    df = handle_categorical_data(df)

    # Standardize / Normalize the dataset
    df = standardize_dataset(df)
    return df

def correlationFigure(featureVariablesMain, targetVariable):
    """
    This fucntion should plot the correlations plot
    :param featureVariablesMain: The entire dataframe
    :param targetVariable: Class Label 'priceSF'
    :return: correlations (correlation coefficients wrt class label)
    """

    # Calculate correlation
    def correlationCalculation(targetVariable, featureVariables, features):
        """
        This function should calculate the correlation coefficients.
        :param targetVariable: Class Label 'priceSF'.
        :param featureVariables: The features variables.
        :param features: column names of the features.
        :return:
        """

        # For maintaining the feature names
        columns = [] 

        # For maintaining the corr values of features with "SalesPrices"
        values = [] 

        # Traverse through all the input features
        for x in features:

            if x is not None:

                # Append the column name
                columns.append(x) 

                # Calculate the correlation
                c = np.corrcoef(
                    featureVariables[x], 
                    featureVariables[targetVariable]
                )

                # Absolute value because important values might miss
                absC = abs(c) 
                values.append(absC[0,1])

        dataDict = {
            'features': columns, 
            'correlation_values': values
        }
        corrValues = pd.DataFrame(dataDict)

        # Sort the value by correlation values
        sortedCorrValues = corrValues.sort_values(by="correlation_values")
        
        # Plot the graph to show the features with their correlation values
        figure, ax = plt.subplots(
                        figsize=(15, 45), 
                        squeeze=True
                    )
        ax.set_title("Correlation Coefficients of Features")
        sns.barplot(
            x=sortedCorrValues.correlation_values, 
            y=sortedCorrValues['features'], ax=ax
        )
        ax.set_ylabel("-----------Corr Coefficients--------->")
        plt.show()
        return sortedCorrValues


    # Make a list of columns
    columns = []
    for x in featureVariablesMain.columns:

        columns.append(x)

    # Remove "SalesPrice" from df
    columns.remove(targetVariable)
    # Compute correlations
    correlations = correlationCalculation(
                        targetVariable, 
                        featureVariablesMain, 
                        columns
                    )
    return correlations


def plot_correlations(dataset):
    """
    Function to plot the correlations
    :param dataset: dataframe
    :return: importantFeatures (Top correlating features)
    """

    target = CLASS_VARIABLE
    targetVariable = dataset[CLASS_VARIABLE].to_frame()
    corrData = correlationFigure(dataset, target)
    importantFeatures = corrData.sort_values(
                            by="correlation_values", 
                            ascending=True
                        ).tail(30)
    return importantFeatures


def feature_engineering(features):
    """
    Function to pick the top features.
    :param features: Features.
    :param target: Class Variable.
    :return features: Reduced Dimentional Features.
    """

    target = features[CLASS_VARIABLE]
    importantFeatures = plot_correlations(features)
    global final_column_name
    final_column_name = importantFeatures['features'].tolist()
    most_correlated_features = features[final_column_name]

    return most_correlated_features, target


def get_rmse(predictions, targets):
    """
    Function to get the Root Mean Squared Error.
    :param predictions: Predicted values.
    :param targets: Actual values.
    :return rmse_val: Root Mean Squared Error.
    """

    differences = predictions - targets                       
    differences_squared = differences ** 2                   
    mean_of_differences_squared = differences_squared.mean()  
    rmse_val = np.sqrt(mean_of_differences_squared)           
    return rmse_val


def evaluation_error(y_test, y_pred):
    """
    This fucntion should return evaluation results
    :param y_test: Actual Set
    :param y_pred: Predicted Set
    :return: Evaluations (mae, mse, mpe, mape, rse)
    """

    mae_sum = 0
    mape_sum = 0
    mpe_sum = 0
    for y_actual, y_prediction in zip(y_test, y_pred):

        mae_sum += abs(y_actual - y_prediction)
        mape_sum += (abs((y_actual - y_prediction)) / y_actual)
        mpe_sum += ((y_actual - y_prediction) / y_actual)

    mae = mae_sum / len(y_test)
    mape = mape_sum / len(y_test)
    mpe = mpe_sum / len(y_test)
    rmse = get_rmse(y_prediction, y_test)
    return mae, mape, mpe, rmse


def fit_xgboost_model(train_X, train_Y, test_X):
    """
    This function returns the regressor fit on the train set.
    :param train_X: Train features.
    :param train_Y: Train labels.
    :return xgboost_classifier: XGBoost Classifier
    """

    train_X = train_X[::].values
    train_Y = train_Y[::].values
    test_X = test_X[::].values
    regressor = XGBRegressor()
    regressor.fit(train_X, train_Y)
    regressor = XGBRegressor(n_estimators=1000)
    regressor.fit(
        train_X, 
        train_Y, 
        early_stopping_rounds=5,          
        eval_set=[(
            train_X, 
            train_Y
        )], 
        verbose=False
    )
    y_pred = regressor.predict(train_X)
    mae, mape, mpe, rmse = evaluation_error(train_Y, y_pred)
    print("\nTrain XGBoost Regreesion Model Evaluations:\n")
    print("Root Mean Squared Error: ", rmse)
    print("Mean Absolute Error: ", mae)
    print("Mean Absolute Percentage Error: ", mape)
    print("Mean Percentage Error: ", mpe)
    y_pred_test = regressor.predict(test_X)
    return y_pred_test


# Main function starts here..
if __name__ == "__main__":

    # Get the data files
    print("******************************************************************")
    file_path = args.data_dir
    train_df, test_df, sample_submission = get_data(file_path)
    print("Data Reading Done.")
    print("******************************************************************")
    
    # Get some information about the dataset.
    get_train_info(train_df)
    print("******************************************************************")

    # Preprocess the dataset
    preprocessed_train_df = preprocess_dataframe(train_df)
    print("Training Set Preprocessed")
    print("******************************************************************")

    # Feature Engineering starts here
    train_X, train_Y = feature_engineering(preprocessed_train_df)
    print("Feature Engineering Done.")
    print("******************************************************************")

    # Test Data Preparation
    test_df[CLASS_VARIABLE] = float(0)
    preprocessed_test_df = preprocess_dataframe(test_df)
    test_X = preprocessed_test_df[final_column_name]
    test_Y = fit_xgboost_model(train_X, train_Y, test_X)
    print("Test Data Prepared for Predictions.")
    print("******************************************************************")

    # Saving the predictions
    test_y_df = pd.DataFrame(data=test_Y, columns=[CLASS_VARIABLE])
    id_df = pd.DataFrame(data=test_df['Id'], columns=['Id'])
    final_submission = pd.concat([id_df, test_y_df], axis=1)
    final_submission.to_csv(PREDICTIONS_FILE, index=False)
    print("Predictions Saved in {}".format(file_path + PREDICTIONS_FILE))
    print("******************************************************************")

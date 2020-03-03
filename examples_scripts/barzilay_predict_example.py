from core.barzilay_predict import barzilayPredict
from sklearn.utils import shuffle
import pandas as pd

'''
In order to predict and reproduce the results in the Barzilay paper, 
we need a method to run the ML script that they used. 
The original script they wrote was designed to be run from the command-line/terminal,
so some adjusting of code was necessary to get this working.
 The core mechanics of the ML script are untouched and are exactly the same as in their
github repo. This file serves as a test to make sure that everything is working properly.

In order to manually install their work as a package that can be interpreted by python,
use the following command:

` pip install git+https://github.com/swansonk14/chemprop.git `

This will make chemprop behave the same as any other package that is install via pip,
then use this script to test that chemprop was properly installed
'''

csv_file = '../dataFiles/logP14k.csv'
raw_df = pd.read_csv(csv_file)
raw_df = shuffle(raw_df)  # Get and Shuffle data

split_index = int(len(raw_df)*0.9)
training_df = raw_df[:split_index]
testing_df = raw_df[split_index:]  # Split data into training and testing

results = barzilayPredict(target_label='kow', df_to_train=training_df, df_to_predict=testing_df,
                          dataset_type='regression', train=True, predict=False)  # Create results object

predicted_results_df = results.predicted_results_df
training_time = results.training_time
predicting_time = results.predicting_time  # Get information from results object

print(predicted_results_df)
print(training_time, predicting_time)

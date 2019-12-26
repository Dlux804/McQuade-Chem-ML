from core.barzilay_predict import barzilayPredict
import pandas as pd

'''
In order to predict and reproduce the results in the Barzilay paper, we need a method to run the ML script that they
used. The original script they wrote was designed to be run from the command-line/terminal, so some adjusting of code
was necessary to get this working. The core mechanics of the ML script are untouched and are exactly the same as in their
github repo. This file serves as a test to make sure that everything is working properly.

In order to get this script up and running, some steps need to be taken. The Barzilay team did not make their work into
a proper package, meaning we can not install their package by pip or conda. In order to manually install their work
as a package that can be interperted by python, follow the following steps.
1) Download their github repo and extract the folder 'chemprops' https://github.com/swansonk14/chemprop
2) Move extracted folder to root directory of project (same location where this file is located)
3) Open terminal and cd to 'chemprops' (located in root directory of current project)
4) Type the command 'pip install -e .' inside of current environment 
    - Highly recommended to use a new virtual/conda environment
5) Delete/move chemprops
This will make chemprops behave the same as any other package that is install via pip, then use this script to test
that chemprops was properly installed
'''

barzilayPredict(target_label='logp', df_to_train=pd.read_csv('dataFiles/Lipophilicity-ID.csv'),
                      df_to_predict=pd.read_csv('dataFiles/Lipophilicity-ID.csv'), dataset_type='regression')
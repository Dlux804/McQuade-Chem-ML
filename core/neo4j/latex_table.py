import pandas as pd


def csv_to_latex(filename):
    df = pd.read_csv(filename)
    df.to_latex

csv_to_latex('node_count.csv')
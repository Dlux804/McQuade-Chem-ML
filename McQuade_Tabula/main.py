import tabula
from os import listdir
import pandas as pd
import numpy as np


def perrys_pdf_to_csv():
    for file in listdir('Perrys_ChemE_Handbook'):
        input_file = 'Perrys_ChemE_Handbook/{}'.format(file)
        output_file = 'Perrys_CSV/{}.csv'.format(file[:len(file) - 4])
        tabula.convert_into(input_file, output_file, output_format='csv', pages='all')

def clean_gathered_str(gathered_str):
    temp_str = gathered_str.split('.0')
    if len(temp_str) > 1:
        if '.' not in temp_str[0]:
            temp_str[0] = temp_str[0]+'.0'
        gathered_str = ''.join(temp_str)
    return gathered_str

def clean_up_densities_data():
    for file in listdir('Perrys_xlsx'):
        input_file = 'Perrys_xlsx/{}'.format(file)
        if file == 'densities.xlsx':
            raw_data = pd.read_excel(input_file)
            clean_data = []
            dicts = []
            for i in range(len(raw_data)):
                row = dict(raw_data.iloc[i,:])
                if not dicts:
                    dicts.append(row)
                elif str(row['Compound Number']) == 'nan':
                    dicts.append(row)
                else:
                    keys = [*dicts[0]]
                    clean_dict = {}
                    for key in keys:
                        gathered_str = []
                        for d in dicts:
                            if str(d[key]) != 'nan':
                                gathered_str.append(str(d[key]))
                        gathered_str = ''.join(gathered_str)
                        gathered_str = clean_gathered_str(gathered_str)
                        clean_dict[key] = gathered_str
                    clean_data.append(clean_dict)
                    dicts = [row]
            clean_data = pd.DataFrame.from_records(clean_data)
            clean_data['Compound Number'] = clean_data['Compound Number'].apply(lambda x: int(float(x)))
            clean_data['Eqn'] = clean_data['Eqn'].apply(lambda x: int(float(x)))
            clean_data.to_csv('Perrys_cleaned_up_data/densities.csv', index=False)


clean_up_densities_data()
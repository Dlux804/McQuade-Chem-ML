import pandas as pd
import numpy as np
from rdkit import Chem 

sorting_data = pd.read_excel('clean_sorting_data.xlsx')
alldata = pd.read_csv("Lipophilicity.csv")

sorted_data = alldata


for i in range(0, len(sorting_data)):
    sm_one = Chem.MolFromSmarts(sorting_data.iloc[i,0])
    sm_two = Chem.MolFromSmarts(sorting_data.iloc[i,2])

    for j in range(0, len(alldata)):
        mol = Chem.MolFromSmiles(alldata.iloc[j,2])
        if mol.HasSubstructMatch(sm_one):
            sorted_data.iloc[j,3] = str(sorted_data.iloc[j,3])+", "+str(sorting_data.iloc[i,0])+":"+str(sorting_data.iloc[i,2])+":Acid:"+str(sorting_data.iloc[i,1])
        if mol.HasSubstructMatch(sm_two):
            sorted_data.iloc[j,3] = str(sorted_data.iloc[j,3])+", "+str(sorting_data.iloc[i,0])+":"+str(sorting_data.iloc[i,2])+":Base:"+str(sorting_data.iloc[i,1])

sorted_data.to_excel("Lipo_Mid_output.xlsx", index=False)
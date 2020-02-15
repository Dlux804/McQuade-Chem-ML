import re
import pandas as pd
from rdkit import Chem 

sorting_data = pd.read_excel('Conj_Acids_Bases.xlsx') #Read conjugate bases and acids
LipoData = pd.read_csv("Lipophilicity.csv") #Read Lipo Data 

LipoData['pKa']=''

for i in range(0, len(sorting_data)):
    sm_one = Chem.MolFromSmarts(sorting_data.iloc[i,0]) #sm_one = mol object of conj. acid
    sm_two = Chem.MolFromSmarts(sorting_data.iloc[i,2]) #sm_two = mol object of conj. base

    for j in range(0, len(LipoData)):
        mol = Chem.MolFromSmiles(LipoData.iloc[j,2]) #mol = mol object of molecule in Lipo data
        if mol.HasSubstructMatch(sm_one): #If sm_one is found in mol
            LipoData.loc[j,'pKa'] = str(LipoData.loc[j,'pKa'])+", "+str(sorting_data.iloc[i,0])+":"+str(sorting_data.iloc[i,2])+":Acid:"+str(sorting_data.iloc[i,1]) #create a line containing all matches, each match containing: Conj acid, conj base, and pka. Also plant in line weather the conj base or acid was found
        if mol.HasSubstructMatch(sm_two): #If sm_two is found in mol
            LipoData.loc[j,'pKa'] = str(LipoData.loc[j,'pKa'])+", "+str(sorting_data.iloc[i,0])+":"+str(sorting_data.iloc[i,2])+":Base:"+str(sorting_data.iloc[i,1]) #create same line as above, except add keyword base instead of acid

LipoData['charged']=''
LipoData['pka-descriptors']=''

for i in range(0, len(LipoData)): 
    molecule = LipoData.loc[i, 'Smiles']
    line = str(LipoData.loc[i, 'pKa']) #Set the keyword line from last code as line
    line = line.split(', ') #Split the line by ','
    line.pop(0) #get rid of the NaN
    ions = ""
    pKas = ""
    check = True 

    if len(line) == 0:
        LipoData.loc[i, 'charged'] = 'False'
    if "+" in molecule: #If there is a + in the ion
        LipoData.loc[i, 'charged'] = 'True' #Set Charged as True
        check = False
    if "-" in molecule: #If there is a - in the ion
        LipoData.loc[i, 'charged'] = 'True' #Set charged as True
        check = False

    '''
    This next part is the chemistry logic of the code. We are looking for molecules that are charged withen the pH range of 7-8. pKa represents
    the point where an ion will either gain a charge or lose it. If the ion in question is a conj base, then if the pka is above 8 the conj acid
    will be found. If the conj base has a pKa less than 7, then it will remain as the conj base. If the ion in question is a conj acid, then if the pKa
    is below 8 the conj base will be found. If the conj acid has a pKa greater than 8, then it will remain as the conj acid. For both conj acids and
    conj bases, if the pka is between 7-8, it is assumed both the conj acid and base will be found in solution.
    '''
    for j in range(0, len(line)):
        values = line[j].split(":")

        if values[2]=='Base' and float(values[3])>=7: #If conj base and pka is above 7
            if float(values[3])>=8: #And if pKa is above 8
                ion = values[0] #Set ion equal to the conj acid
            else:
                ion = values[0]+":"+values[1] #If pka is between 7-8, set ion as both the conj acid and conj base
        if values[2]=='Base' and float(values[3])<7: #If conj base and pka is below 7
            ion = values[1] #Set ion equal to the conj base

        if values[2]=='Acid' and float(values[3])<=8: #If conj acid and pka is below 8
            if float(values[3])<=7: #And if pka is below 7
                ion = values[1] #Set ion equal to the conj base
            else:
                ion = values[0]+":"+values[1] #If pka is between 7-8, set ion as both the conj acid and base
        if values[2]=='Acid' and float(values[3])>8: #if conj acid has a pka above 8
                ion = values[0] #Set ion equal to the conj acid

        if "+" in ion: #If there is a + in the ion
            LipoData.loc[i, 'charged'] = 'True' #Set Charged as True
            check = False
        if "-" in ion: #If there is a - in the ion
            LipoData.loc[i, 'charged'] = 'True' #Set charged as True
            check = False
        if check == True: #If none of the ions are charged
            LipoData.loc[i, 'charged'] = 'False' #Set charged as False
        
        if ions == "":
            ions = ion+":"+str(values[3])
        else:
            ions = ions+", "+ion+":"+str(values[3]) #Set the ions as all the ions found, plus tack on the pka for each pka-site
        LipoData.loc[i, 'pka-descriptors'] = ions


LipoData = LipoData.drop(['pKa'], axis=1)
LipoData.to_excel('charge_predictions.xlsx', index=False) 

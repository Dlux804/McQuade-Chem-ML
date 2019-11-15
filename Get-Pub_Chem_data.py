import urllib.request
import gzip
import shutil
import pybel
import os
import pandas as pd
import time

#Change directory to folder to contain .sdf files

#num_of_files = int(139275000/25000)
num_of_files = 500
counter = 0
sum_of_times = 0

print('\nThere are '+str(num_of_files)+' files to extract...')
#for i in range(1, num_of_files): #This is for all files on the database
for i in range(1,500): #This is to extract how ever many files needed

    os.chdir(r'C:\Users\andre\Desktop\Work\scripts\Scrapy\Pub_Chem\Pub_Chem_csvs')

    num_of_files = num_of_files - i #This is a override, since 'only' looking at 1000 files

    start_time = time.time()
    counter = counter + 1

    Upper_Bound = 25000*i #The files use the upper and lower bounds as reference, the code will calculate the upper and lower bounds
    Lower_Bound = Upper_Bound-24999
    Upper_Bound, Lower_Bound = str(Upper_Bound), str(Lower_Bound)

    while len(Lower_Bound) < 9:
        Lower_Bound = '0' + Lower_Bound

    while len(Upper_Bound) < 9:
        Upper_Bound = '0' + Upper_Bound

    file = 'Compound_'+Lower_Bound+'_'+Upper_Bound 

    link = 'ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/CURRENT-Full/SDF/'+file+'.sdf.gz' #Get the link based on upper and lower bounds

    urllib.request.urlretrieve(link, file+'.sdf.gz') #Download the compressed file to a folder

    with gzip.open(file+'.sdf.gz', 'rb') as f_in: #This will decompress and save the file in the same folder
        with open(file+'.sdf', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(file+'.sdf.gz') #Delete the compressed file

    moleculeData = []
    for mol in pybel.readfile("sdf", file+".sdf"):
       molDict = dict(zip(mol.data.keys(), mol.data.values()))
       moleculeData.append(molDict)
    
    moleculeData = pd.DataFrame(moleculeData)
    moleculeData.to_csv(file+'.csv', index=False)
    os.remove(file+'.sdf')

    os.chdir(r'F:\pub_chem_csvs')
    moleculeData.to_csv(file+'.csv', index=False)
    print('\ni = '+str(i)+' | Created File: '+file+'.csv')
    
    checker_time = time.time() - start_time
    sum_of_times = sum_of_times + checker_time
    ETA = (sum_of_times/counter) * num_of_files
    if ETA/86400 > 1:
        print('Estimated Time Needed to Complete: '+str(round(ETA/86400, 2))+' days')
    elif ETA/3600 > 1:
        print('Estimated Time Needed to Complete: '+str(round(ETA/3600, 2))+' hrs')
    elif ETA/60 > 1:
        print('Estimated Time Needed to Complete: '+str(round(ETA/60, 2))+' mins')
    else:
        print('Estimated Time Needed to Complete: '+str(round(ETA, 2))+' secs')
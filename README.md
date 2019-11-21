# McQuade-Chem-ML
Development of easy to use and reproducible ML scripts for chemistry.  

## Data Sets
Currently we are using three datasets from MoleculeNet.ai: Lipophilicity, FreeSOLV and ESOL.

## Available Models
Our program supports random forest (RF), gradient decent boost (GDB), support vector machines (SVM).  


### Dependencies
We should all be using the same conda evironment so that we do not run into the issue
of "Well it works on my machine".  To do this, we will host a .yml file for the shared
environment on our repo (mlapp.yml).

1. Create an conda virual environment from the mlapp.yml file
 ```conda env create -f mlapp.yml```
 2. Update the virtual environment as necessary using ```conda install```
 3. Update the mlapp.yml file using ```conda env export > mlapp.yml```. Make sure that you add the 
 mlapp.yml file to git, if it not already being watched.

    **Note:** Sometimes packages cannot be installed from conda, such as descriptastorus.
    If this is the case, you may need to use pip to install from a github link.
    See the mlapp.yml file for an example (descriptastorus) for an example of how to account for this
    in the mlapp.yml file.  
 4. Commit your changes, which include the mlapp.yml file. ```git commit -m "your commit message here"```
 
 
# Workflow
![Alt text](./hte-models-overview.svg)

# MLKG

![Alt text](graphics/mlkg_landing_fig.png)

A Python pipeline for creating basic machine learning models for chemical property predictions.
The models are used to populate a Neo4j property graph database.  
## Data Sets
Currently we are using three datasets from MoleculeNet.ai: Lipophilicity, FreeSOLV and ESOL
 and three datasets retreived from tutorials and publications: LogP14k, jak2-pIC50, and flashpoint.  
 The CSVs for the datasets can be found in `/dataFiles`

## Available Models
Our program supports Scikit-Learn algorithms for random forest (RF), gradient decent boost (GDB),
 support vector machines (SVM), Adaboost, and k-nearest neightbors (KNN).   We implement dense neural networks
 with Tensorflow and the Keras API.

### Getting Set Up
The following will serve as a guide for getting our models running on your computer. Users should also install
Neo4j Desktop if they wish to use the graph features. 

**NOTE:** YOU MUST HAVE GIT, PYCHARM, AND ANACONDA INSTALLED IN ORDER TO USE THE FOLLOWING INSTRUCTIONS. However, it is possible to run our models using different programs (GIT is required to clone our repository).

Git download link: https://git-scm.com/downloads

Pycharm download link (windows): https://www.jetbrains.com/pycharm/download/#section=windows

Anaconda download link: https://www.anaconda.com/distribution/

 With Git, Pycharm, and Anaconda installed, use Pycharm's "get from version control" option to clone our repository (you can copy/paste this link: https://github.com/Dlux804/McQuade-Chem-ML).

 ![Alt text](graphics/Getting-set-up-picture.png)

### Dependencies
We  host a .yml file for the conda environment used in our work (mlapp2.yml).

1. Create an conda virtual environment in the Anaconda prompt from the mlapp.yml file
 ```conda env create -f env_init\mlapp.yml```
 Note: On windows, you will have you comment out "-gunicorn" in the mlapp.yml file.
 
 ![Alt text](graphics/Dependecies-step-1-picture2.png)
 
 ![Alt text](graphics/Dependecies-step-1-picture.png)
 
 2. In Pycharm, go to the bottom right and hit the interpreter button. Select "Add interpreter".
 
 
 ![Alt text](graphics/Dependecies-step-2-picture.png)
 
 3. Navigate to "Conda Environment" and select "Existing environment". The mlapp\python.exe environment should be located in the Anaconda3\envs folder. Select this interpreter and check "Make available to all projects". Hit OK. You should now be able to run our code.
 
  ![Alt text](graphics/Dependecies-step-3-picture.png)
 
 4. Update the virtual environment as necessary using ```conda install```
 5. Update the mlapp.yml file using ```conda env export > mlapp.yml --no-builds --from-history```. Make sure that you add the 
 mlapp.yml file to git, if it not already being watched.

    **Note:** Sometimes packages cannot be installed from conda, such as descriptastorus.
    If this is the case, you may need to use pip to install from a github link.
    See the mlapp.yml file for an example (descriptastorus) for an example of how to account for this
    in the mlapp.yml file.  
    ```
    - pip:
        - "git+git://github.com/bp-kelley/descriptastorus.git#egg=descriptastorus"
    ```
 6. Commit your changes, which include the mlapp.yml file. ```git commit -m "your commit message here"```
 
 7. To use DeepChem, revert scikit-learn to version 0.22.0 by using the command ```pip install scikit-learn==0.22.0```

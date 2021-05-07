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

## Getting Started
The following will serve as a guide for getting our models running on your computer. Users should also install
Neo4j Desktop if they wish to use the graph features. 

**NOTE:** YOU MUST HAVE GIT, PYCHARM, AND ANACONDA INSTALLED IN ORDER TO USE THE FOLLOWING INSTRUCTIONS. However, it is possible to run our models using different programs (GIT is required to clone our repository).

Git download link: https://git-scm.com/downloads

Pycharm download link (windows): https://www.jetbrains.com/pycharm/download/#section=windows

Anaconda download link: https://www.anaconda.com/distribution/

 With Git, Pycharm, and Anaconda installed, use Pycharm's "get from version control" option to clone our repository (you can copy/paste this link: https://github.com/Dlux804/McQuade-Chem-ML).

 ![Alt text](graphics/Getting-set-up-picture.png)

### Important Dependencies
- Python 3.7 or 3.6
- rdkit=2020.09.1.0
- py2neo=4.2.0
- scikit-learn=0.23.2
- descriptastorus=2.2.0
- scikit-optimize=0.8.1
- tensorflow-gpu==1.15.0



 ### Enviroment Set Up

- Create a conda virtual environment in Anaconda Prompt for Windows, Terminal for Linux, from the `mlapp.yml` file and the `requirements.txt` in our `env_init` folder. This
should work on both Windows and Ubuntu.

    ```conda env create -f mlapp.yml```
    
    ```conda activate mlapp```
    
    ```pip install -r requirements.txt```
 
 ![Alt text](graphics/Dependecies-step-1-picture2.png)
 
 
 - In Pycharm, go to the bottom right and hit the interpreter button. Select "Add interpreter".
 
 
 ![Alt text](graphics/Dependecies-step-2-picture.png)
 
- Navigate to "Conda Environment" and select "Existing environment". 
The mlapp\python.exe environment should be located in the Anaconda3\envs folder. Select this interpreter and check "Make available to all projects". Hit OK. You should now be able to run our code.
  
- Note, if you get an error while installing the environment in the Anaconda Prompt, try running Anaconda Prompt as Admin
   
### Neo4j
To output models into Neo4j, first you will need to create a local Database
 1. Open Neo4j and add a `Local DBMS`
 2. Download the APOC and Graph Data Science (GDS) library plugins. If you're using the latest edition of Neo4j, GDS will be pre-installed for you.

### MySql
MySql is used to store featurized datasets and helps saves time. The instructions to install MySql for both Windows and Linux 
can be found at https://dev.mysql.com/doc/mysql-getting-started/en/. If installing on Windows, the Server only version is highly recommend. 
Further instructions for windows can be found at https://phoenixnap.com/kb/install-mysql-on-windows.

A new user and data table should be created. The MySQL server can be access in Windows by opening the command prompt and typing in

```cd Program Files\MySQL\MySQL Server 8.0\bin```

```mysql -u root -p```

For Linux, open the terminal and type in

```mysql -u root -p```

Then type in the password that was used to setup the root user. Then instructions to create a new database and
user can be found here https://matomo.org/faq/how-to-install/faq_23484/. Then enter you parameters in the line in main.py 

```model.connect_mysql(user='user', password='dolphin', host='localhost', database='featurized_datasets', initialize_all_data=False)```

To the parameters used to setup the database. 

If you do not wish to install MySql or DataGrip, then you can disable featurizing from MySql by setting 
```model.featurize(retrieve_from_mysql=True)``` to ```model.featurize(retrieve_from_mysql=False)``` in main.py. MySQL is
not required, and only helps to speed up featurization of data. Also, an IDE such as DataGrip can be used to see the 
data tables being created by main.py. 

 
## Run the Pipeline
1. Start your Neo4j local DBMS and let in run in the background
2. If using MySQL, start MySQL database
3. Run main.py
4. Enjoy!! 
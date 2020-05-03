import ast
import os
import timeit
import pandas as pd

import py2neo
from py2neo import Graph
import concurrent.futures as cf
from rdkit.Chem import MolToSmiles, MolFromSmiles
from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs, clean_up_checker_files

reaction_string = """

UNWIND $parameters as row
MERGE (rxn:reaction {reaction_smiles: row.reaction_smiles})
ON CREATE SET rxn.sources = row.sources, rxn.insert_stages = row.stages

FOREACH (reactant in row.reactants | 
         MERGE (com:compound {smiles: reactant.smiles}) 
         ON CREATE SET com.chemical_names = reactant.chemical_names, com.appearances = reactant.appearances,
                       com.inchi = reactant.inchi
         MERGE (com)-[:reacts]->(rxn)
        )
        
FOREACH (solvent in row.solvents | 
         MERGE (com:compound {smiles: solvent.smiles})
         ON CREATE SET com.chemical_names = solvent.chemical_names, com.appearances = solvent.appearances,
                       com.inchi = solvent.inchi 
         MERGE (com)-[:solvent_for]->(rxn)
        )
        
FOREACH (catalyst in row.catalyst | 
         MERGE (com:compound {smiles: catalyst.smiles})
         ON CREATE SET com.chemical_names = catalyst.chemical_names, com.appearances = catalyst.appearances,
                       com.inchi = catalyst.inchi
         MERGE (com)-[:catalysis]->(rxn)
        )

FOREACH (product in row.products | 
         MERGE (com:compound {smiles: product.smiles})
         ON CREATE SET com.chemical_names = product.chemical_names, com.appearances = product.appearances,
                       com.inchi = product.inchi
         MERGE (rxn)-[:produces]->(com)
        )
        
"""

def check_for_constraint():

    compound_constraint_check_string = """
        CREATE CONSTRAINT unique_smiles
        ON (n:compound)
        ASSERT n.smiles IS UNIQUE
    """
    reaction_constraint_check_string = """
            CREATE CONSTRAINT unique_reaction_smiles
            ON (n:reaction)
            ASSERT n.reaction_smiles IS UNIQUE
        """
    try:
        graph = Graph()
        tx = graph.begin(autocommit=True)
        tx.evaluate(compound_constraint_check_string)
        tx = graph.begin(autocommit=True)
        tx.evaluate(reaction_constraint_check_string)
    except py2neo.database.ClientError:
        pass


def clean_up_compound(compound):
    identifiers = compound['identifiers']
    check = False
    for identifier in identifiers:
        id_key = identifier.split(' = ')[0]
        id_value = identifier.split(' = ')[1]
        if id_key == 'smiles':
            smiles = id_value
            mol = MolFromSmiles(smiles)
            if mol is not None:
                smiles = MolToSmiles(mol)
                compound['smiles'] = smiles
                check = True
        else:
            inchi = id_value
            compound['inchi'] = inchi

    compound.pop('identifiers')
    if check:
        return compound


def gather_reactions(reaction_row):
    dict_row = dict(reaction_row)
    compound_labels = ['reactants', 'products', 'catalyst', 'solvents']
    for item in dict_row:
        if item in compound_labels:
            new_compounds = []
            compounds = ast.literal_eval(dict_row[item])
            for compound in compounds:
                new_compound = clean_up_compound(compound)
                if new_compound is not None:
                    new_compounds.append(new_compound)
            dict_row[item] = new_compounds
    return dict_row

def __main__(file):


    file_data = pd.read_csv(file)
    graph = Graph()

    print(f"There are {len(file_data)} reactions in file")

    reactions = []

    with cf.ThreadPoolExecutor() as executor:

        results = []
        for index, row in file_data.iterrows():
            results.append(executor.submit(gather_reactions, row))

        for f in cf.as_completed(results):
            reactions.append(f.result())

    file_data = pd.DataFrame.from_records(reactions)

    reactions = []
    for index, row in file_data.iterrows():
        reactions.append(dict(row))
        if index % 20000 == 0 and index > 0:
            tx = graph.begin(autocommit=True)
            tx.evaluate(reaction_string, parameters={"parameters": reactions})

    tx = graph.begin(autocommit=True)
    tx.evaluate(reaction_string, parameters={"parameters": reactions})


if __name__ == "__main__":

    US_patents_directory = 'C:/Users/User/Desktop/5104873'
    # US_grants_directory_to_csvs(US_patents_directory)  # Create csv directories
    clean_up_checker_files(US_patents_directory)

    main_timer = timeit.default_timer()
    check_for_constraint()
    for main_directory in os.listdir(US_patents_directory):
        main_directory = US_patents_directory + '/' + main_directory
        for directory in os.listdir(main_directory):
            if directory[-4:] == '_csv':
                directory = main_directory + '/' + directory
                number_of_files = len(os.listdir(directory))
                i = 0
                for file in os.listdir(directory):
                    file = directory + '/' + file

                    file_split = file.split('.')
                    file_split = file_split[len(file_split) - 1]

                    if not os.path.exists(file + '.checker') and file_split != 'checker':
                        print("---------------------------------------")
                        print(f"Working in directory {directory}\n"
                              f"There are {number_of_files - i} files remaining")
                        start_timer = timeit.default_timer()
                        try:
                            __main__(file)
                            open(file + ".checker", "a").close()
                            time_needed = round((timeit.default_timer() - start_timer), 2)
                            print(f"Time Needed for file {time_needed} seconds")
                        except pd.errors.EmptyDataError:
                            open(file + ".checker", "a").close()
                    i = i + 1
    time_needed_minutes = round((timeit.default_timer() - main_timer)/60, 2)
    time_needed_hours = round(time_needed_minutes/60, 2)
    print("---------------------------------------")
    print(f"Time Needed to insert all files into Neo4j {time_needed_minutes} minutes,"
          f"{time_needed_hours} hours")

import ast
import os
import timeit
import pandas as pd
import tqdm

import py2neo
from py2neo import Graph
from rdkit.Chem import MolToSmiles, MolFromSmiles, rdChemReactions
from rdkit.Chem.Descriptors import MolWt
import concurrent.futures as cf
from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs
from Neo4j.US_patents.backends import clean_up_checker_files, save_reaction_image, get_fragments, get_file_location

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

'''
The reaction string below is the query that is feed to Neo4j that will insert all the reactions into neo4j. The
UNWIND parameter allows neo4j to literate over all rows in a list in a very effective manner compared to traditional
insert methods. This is likely not even the fastest way to insert all the data into Neo4j, but at our scale this
is fast enough. The slowest parts are are functions that need to happen in python, such as: converting non-con smiles
to con smiles, getting reaction smarts, and finding the fragments in the reaction smarts.
APOC will be faster if we ever get to the point where we need to insert more than 50 GBs of data
'''

reaction_query = """

UNWIND $parameters as row
MERGE (rxn:reaction {reaction_smiles: row.reaction_smiles})
ON CREATE SET rxn.reactant_fragments = row.reactant_fragments, 
              rxn.product_fragments = row.product_fragments, rxn.sources = row.sources, rxn.insert_stages = row.stages,
              rxn.reaction_image_location = row.reaction_image_location

FOREACH (reactant in row.reactants | 
         MERGE (com:compound {smiles: reactant.smiles}) 
         ON CREATE SET com.chemical_names = reactant.chemical_names, com.appearances = reactant.appearances,
                       com.inchi = reactant.inchi, com.molar_mass = reactant.molwt, 
                       com.functional_groups = reactant.functional_groups
         MERGE (com)-[:reacts]->(rxn)
        )

FOREACH (solvent in row.solvents | 
         MERGE (com:compound {smiles: solvent.smiles})
         ON CREATE SET com.chemical_names = solvent.chemical_names, com.appearances = solvent.appearances,
                       com.inchi = solvent.inchi, com.molar_mass = solvent.molwt, 
                       com.functional_groups = solvent.functional_groups
         MERGE (com)-[:solvent_for]->(rxn)
        )

FOREACH (catalyst in row.catalyst | 
         MERGE (com:compound {smiles: catalyst.smiles})
         ON CREATE SET com.chemical_names = catalyst.chemical_names, com.appearances = catalyst.appearances,
                       com.inchi = catalyst.inchi, com.molar_mass = catalyst.molwt, 
                       com.functional_groups = catalyst.functional_groups
         MERGE (com)-[:catalysis]->(rxn)
        )

FOREACH (product in row.products | 
         MERGE (com:compound {smiles: product.smiles})
         ON CREATE SET com.chemical_names = product.chemical_names, com.appearances = product.appearances,
                       com.inchi = product.inchi, com.molar_mass = product.molwt, 
                       com.functional_groups = product.functional_groups
         MERGE (rxn)-[:produces]->(com)
        )   
"""

'''
This function will check and make sure that constraints are on the graph. This is important because constraints allow
nodes to replace IDs with, well the constraint label. This allows Neo4j to quickly merge the different nodes. 
Constraints plus UNWIND allow for some really impressive speed up times in inserting data into Neo4j.
'''


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


"""
This function will clean up the properties of the various compounds in the file, as well as convert non-con smiles to
con smiles.
"""


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
                compound['molwt'] = MolWt(mol)
                if insert_compounds_with_functional_groups:
                    compound['functional_groups'] = get_fragments(smiles, fragments_df=fragments_df)
        else:
            inchi = id_value
            compound['inchi'] = inchi

    compound.pop('identifiers')
    if check:
        return compound
    else:
        return None


def clean_up_compounds(compounds):
    new_compounds = []
    compounds = ast.literal_eval(compounds)
    for compound in compounds:
        new_compound = clean_up_compound(compound)
        if new_compound is not None:
            new_compounds.append(new_compound)
    return new_compounds


"""
Logic to get reaction smarts from reactions, and get the what fragments change between the reactants and products.
The longest step in the process of inserting data into the database. 
"""


def get_new_reaction_smiles(reaction_smiles):
    if len(reaction_smiles.split('|')[0]) > 1:
        reaction_smiles = reaction_smiles.split('|')[0]

    rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles)
    new_smiles = rdChemReactions.ReactionToSmarts(rxn)
    return new_smiles


def __aw__(df_column, function, **props):
    new_df_column = df_column.apply(function, **props)
    return new_df_column


def parallel_apply(df_column, function, **props):
    steps = len(df_column) / number_of_cpus
    mid_dfs = []
    for x in range(number_of_cpus):
        if x == number_of_cpus - 1:
            mid_dfs.append(df_column.iloc[int(steps * x):])
        else:
            mid_dfs.append(df_column.iloc[int(steps * x):int(steps * (x + 1))])

    main_df = None
    with cf.ProcessPoolExecutor(max_workers=number_of_cpus) as executor:

        results = []
        for mid_df in mid_dfs:
            results.append(executor.submit(__aw__, mid_df, function, **props))

        if loading_bars:
            for f in tqdm.tqdm(cf.as_completed(results), total=number_of_cpus):
                if main_df is None:
                    main_df = f.result()
                else:
                    main_df = main_df.append(f.result())
        else:
            for f in cf.as_completed(results):
                if main_df is None:
                    main_df = f.result()
                else:
                    main_df = main_df.append(f.result())

    return main_df


"""
This is the main function that wraps everything up. First the graph is spun up, and the data from the csv file is read.
Then using parallelization, all the rows in the dataframe are cleaned up and added to a list. Then the list is feed to 
neo4j where the reactions are merged to the graph.
"""


def __main__(working_file):
    start_timer = timeit.default_timer()

    file_data = pd.read_csv(working_file)
    graph = Graph()

    print(f"There are {len(file_data)} reactions in file")

    file_data['reaction_smiles'] = parallel_apply(file_data['reaction_smiles'], get_new_reaction_smiles)
    file_data['reactants'] = parallel_apply(file_data['reactants'], clean_up_compounds)
    file_data['products'] = parallel_apply(file_data['products'], clean_up_compounds)
    file_data['solvents'] = parallel_apply(file_data['solvents'], clean_up_compounds)
    file_data['catalyst'] = parallel_apply(file_data['catalyst'], clean_up_compounds)

    if save_reaction_images:
        file_data['reaction_image_location'] = parallel_apply(file_data['reaction_smiles'], save_reaction_image,
                                                              directory_location=reaction_images_directory)

    reactions = []
    for index, row in file_data.iterrows():
        reactions.append(dict(row))
        if index % 20000 == 0 and index > 0:
            tx = graph.begin(autocommit=True)
            tx.evaluate(reaction_query, parameters={"parameters": reactions})

    tx = graph.begin(autocommit=True)
    tx.evaluate(reaction_query, parameters={"parameters": reactions})

    time_needed = round((timeit.default_timer() - start_timer), 2)
    print(f"Time Needed for file {time_needed} seconds")

    if log_time_needed:
        time_df = pd.read_csv('Time_df.csv')
        time_df = time_df.append({'Number of Reactions': len(file_data), 'Time Needed (s)': time_needed},
                                 ignore_index=True)
        time_df.to_csv('Time_df.csv', index=False)


if __name__ == "__main__":

    number_of_cpus = 4
    US_patents_directory = '/home/user/Desktop/5104873'
    fragments_df = pd.read_csv(get_file_location() + '/datafiles/Function-Groups-SMARTS.csv')
    covert_xml_to_csv = False
    clean_checker_files = False
    insert_compounds_with_functional_groups = True
    log_time_needed = True
    loading_bars = False
    save_reaction_images = False
    reaction_images_directory = None

    if not os.path.exists('Time_df.csv') and log_time_needed:
        df = pd.DataFrame(columns=['Number of Reactions', 'Time Needed (s)'])
        df.to_csv('Time_df.csv', index=False)

    if covert_xml_to_csv:
        US_grants_directory_to_csvs(US_patents_directory)
    if clean_checker_files:
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
                        try:
                            __main__(file)
                            open(file + ".checker", "a").close()
                        except pd.errors.EmptyDataError:
                            open(file + ".checker", "a").close()
                    i = i + 1
    time_needed_minutes = round((timeit.default_timer() - main_timer) / 60, 2)
    time_needed_hours = round(time_needed_minutes / 60, 2)
    print("---------------------------------------")
    print(f"Time Needed to insert all files into Neo4j, time needed: {time_needed_minutes} minutes, "
          f"{time_needed_hours} hours")

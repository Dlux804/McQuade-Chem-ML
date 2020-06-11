import ast
import os
import timeit
import tqdm
import multiprocessing
import concurrent.futures as cf
import pandas as pd

import py2neo
from py2neo import Graph
from rdkit.Chem import MolToSmiles, MolFromSmiles, rdChemReactions
from rdkit.Chem.Descriptors import MolWt
from Neo4j.US_patents.US_patents_xml_to_csv import US_grants_directory_to_csvs
from Neo4j.US_patents.backends import clean_up_checker_files, get_functional_groups, \
    get_file_location, map_rxn_functional_groups

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class InitPatentsIntoNeo4j:

    def __init__(self, patents_directory=None, number_of_cups=multiprocessing.cpu_count(),
                 convert_xml_to_csv=False, clean_checker_files=False, insert_compounds_with_functional_groups=False,
                 insert_change_in_functional_groups=False, loading_bars=True):

        if patents_directory is None:
            raise Exception("Directory for US patents must be specified")

        self.US_patents_directory = patents_directory
        self.number_of_cpus = number_of_cups
        self.covert_xml_to_csv = convert_xml_to_csv
        self.clean_checker_files = clean_checker_files
        self.insert_compounds_with_functional_groups = insert_compounds_with_functional_groups
        self.insert_change_in_functional_groups = insert_change_in_functional_groups
        self.loading_bars = loading_bars

        self.fragments_df = pd.read_csv(get_file_location() + '/datafiles/Function-Groups-SMARTS.csv')
        self.__main__()

    def __main__(self):
        """
        The main function of this script. Will convert xml files to csv and/or delete checker files if specified. Then
        will make sure that the proper constraints are applied to be able to index using smiles.
        """

        if self.covert_xml_to_csv:
            US_grants_directory_to_csvs(self.US_patents_directory)
        if self.clean_checker_files:
            clean_up_checker_files(self.US_patents_directory)

        main_timer = timeit.default_timer()
        self.__check_for_constraint__()
        for main_directory in os.listdir(self.US_patents_directory):
            main_directory = self.US_patents_directory + '/' + main_directory
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
                                self.__run_file__(file)
                                open(file + ".checker", "a").close()
                            except pd.errors.EmptyDataError:
                                open(file + ".checker", "a").close()
                        i = i + 1
        time_needed_minutes = round((timeit.default_timer() - main_timer) / 60, 2)
        time_needed_hours = round(time_needed_minutes / 60, 2)
        print("---------------------------------------")
        print(f"Time Needed to insert all files into Neo4j, time needed: {time_needed_minutes} minutes, "
              f"{time_needed_hours} hours")

    @staticmethod
    def __check_for_constraint__():
        """
        This function will check and make sure that constraints are on the graph. This is important because
        constraints allow nodes to replace IDs with, well the constraint label. This allows Neo4j to quickly merge
        the different nodes. Constraints plus UNWIND allow for some really impressive speed up times in inserting
        data into Neo4j.
        """

        compound_constraint_check_string = """
                CREATE CONSTRAINT unique_smiles
                ON (n:Compound)
                ASSERT n.smiles IS UNIQUE
            """
        reaction_constraint_check_string = """
                    CREATE CONSTRAINT unique_reactionSmiles
                    ON (n:Reaction)
                    ASSERT n.reactionSmiles IS UNIQUE
                """
        try:
            tx = graph.begin(autocommit=True)
            tx.evaluate(compound_constraint_check_string)
            tx = graph.begin(autocommit=True)
            tx.evaluate(reaction_constraint_check_string)
        except py2neo.database.ClientError:
            pass

    def __run_file__(self, working_file):
        """
        This is the main function that wraps everything up. First the graph is spun up, and the data from the csv
        file is read. Then using parallelization, all the rows in the dataframe are cleaned up and added to a list.
        Then the list is feed to neo4j where the reactions are merged to the graph.
        """

        start_timer = timeit.default_timer()

        file_data = pd.read_csv(working_file)

        print(f"There are {len(file_data)} reactions in file")

        file_data['reaction_smiles'] = self.parallel_apply(file_data['reaction_smiles'], self.get_new_reaction_smiles)
        file_data['reactants'] = self.parallel_apply(file_data['reactants'], self.clean_up_compounds)
        file_data['products'] = self.parallel_apply(file_data['products'], self.clean_up_compounds)
        file_data['solvents'] = self.parallel_apply(file_data['solvents'], self.clean_up_compounds)
        file_data['catalyst'] = self.parallel_apply(file_data['catalyst'], self.clean_up_compounds)

        if self.insert_change_in_functional_groups:
            file_data['change_in_functional_groups'] = self.parallel_apply(file_data['reaction_smiles'],
                                                                           map_rxn_functional_groups)

        file_data['patent_index'] = self.parallel_apply(file_data['sources'], self.get_index)
        file_data['reaction_details'] = self.parallel_apply(file_data['sources'], self.get_reaction_details)

        file_data = file_data.drop(['sources', 'stages'], axis=1)

        reactions = []
        for index, row in file_data.iterrows():
            reactions.append(dict(row))
            if index % 20000 == 0 and index > 0:
                tx = graph.begin(autocommit=True)
                tx.evaluate(self.__get_reaction_query__(), parameters={"parameters": reactions})

        tx = graph.begin(autocommit=True)
        tx.evaluate(self.__get_reaction_query__(), parameters={"parameters": reactions})

        time_needed = round((timeit.default_timer() - start_timer), 2)
        print(f"Time Needed for file {time_needed} seconds")

    @staticmethod
    def __aw__(df_column, function, **props):
        """
        Wrapper function for parallel apply. Actual runs the pandas.apply on an individual CPU.
        """

        new_df_column = df_column.apply(function, **props)
        return new_df_column

    def parallel_apply(self, df_column, function, **props):
        """
        This function will run pandas.apply in parallel depending on the number of CPUS the user specifies.
        """

        steps = len(df_column) / self.number_of_cpus
        mid_dfs = []
        for x in range(self.number_of_cpus):
            if x == self.number_of_cpus - 1:
                mid_dfs.append(df_column.iloc[int(steps * x):])
            else:
                mid_dfs.append(df_column.iloc[int(steps * x):int(steps * (x + 1))])

        main_df = None
        with cf.ProcessPoolExecutor(max_workers=self.number_of_cpus) as executor:

            results = []
            for mid_df in mid_dfs:
                results.append(executor.submit(self.__aw__, mid_df, function, **props))

            if self.loading_bars:
                for f in tqdm.tqdm(cf.as_completed(results), total=self.number_of_cpus):
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

    @staticmethod
    def get_new_reaction_smiles(reaction_smiles):
        """
        Some of the reactions have weird formats, this function will cleanup malformed reactions smiles as well
        as return con reaction-smiles/smarts
        """

        if len(reaction_smiles.split('|')[0]) > 1:
            reaction_smiles = reaction_smiles.split('|')[0]

        rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles)
        new_smiles = rdChemReactions.ReactionToSmarts(rxn)
        return new_smiles

    def clean_up_compounds(self, compounds):
        """
        Run a loop over clean_up_compound for a list of compounds
        """

        new_compounds = []
        compounds = ast.literal_eval(compounds)
        for compound in compounds:
            new_compound = self.clean_up_compound(compound)
            if new_compound is not None:
                new_compounds.append(new_compound)
        return new_compounds

    def clean_up_compound(self, compound):
        """
        This function will clean up the properties of the various compounds in the file, as well as convert non-con
        smiles to con smiles.
        """

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
                    if self.insert_compounds_with_functional_groups:
                        compound['functional_groups'] = get_functional_groups(smiles, fragments_df=self.fragments_df)
            else:
                inchi = id_value
                compound['inchi'] = inchi

        chemical_names = compound['chemical_names']
        compound['chemical_name'] = chemical_names[len(chemical_names) - 1]

        appearances = compound['appearances']
        if appearances:
            compound['appearance'] = appearances[0]
        else:
            compound['appearance'] = None

        compound.pop('identifiers')
        compound.pop('chemical_names')
        compound.pop('appearances')
        if check:
            return compound
        else:
            return None

    @staticmethod
    def get_index(source):
        source = ast.literal_eval(source)
        return source[0]

    @staticmethod
    def get_reaction_details(source):
        source = ast.literal_eval(source)
        source.pop(0)
        return " ".join(source)

    @staticmethod
    def __get_reaction_query__():
        """
        The reaction string below is the query that is feed to Neo4j that will insert all the reactions into neo4j.
        The UNWIND parameter allows neo4j to literate over all rows in a list in a very effective manner compared to
        traditional insert methods. This is likely not even the fastest way to insert all the data into Neo4j,
        but at our scale this is fast enough. The slowest parts are are functions that need to happen in python,
        such as: converting non-con smiles to con smiles, getting reaction smarts, and finding the fragments in the
        reaction smarts. APOC will be faster if we ever get to the point where we need to insert more than 50 GBs of
        data
        """

        reaction_query = """
        
        UNWIND $parameters as row
        MERGE (rxn:Reaction {reactionSmiles: row.reaction_smiles})
        ON CREATE SET rxn.reactionDetails = row.reaction_details,
                      rxn.changeInFunctionalGroups = row.change_in_functional_groups,
                      rxn.patentIndex = row.patent_index
        
        FOREACH (reactant in row.reactants | 
                 MERGE (com:Compound {smiles: reactant.smiles}) 
                 ON CREATE SET com.chemicalName = reactant.chemical_name, com.appearance = reactant.appearance,
                               com.inchi = reactant.inchi, com.molarMass = reactant.molwt, 
                               com.functionalGroups = reactant.functional_groups
                 MERGE (com)-[:reacts]->(rxn)
                )
        
        FOREACH (solvent in row.solvents | 
                 MERGE (com:Compound {smiles: solvent.smiles})
                 ON CREATE SET com.chemicalName = solvent.chemical_name, com.appearance = solvent.appearance,
                               com.inchi = solvent.inchi, com.molarMass = solvent.molwt, 
                               com.functionalGroups = solvent.functional_groups
                 MERGE (com)-[:solvates]->(rxn)
                )
        
        FOREACH (catalyst in row.catalyst | 
                 MERGE (com:Compound {smiles: catalyst.smiles})
                 ON CREATE SET com.chemicalName = catalyst.chemical_name, com.appearance = catalyst.appearance,
                               com.inchi = catalyst.inchi, com.molarMass = catalyst.molwt, 
                               com.functionalGroups = catalyst.functional_groups
                 MERGE (com)-[:catalyzes]->(rxn)
                )
        
        FOREACH (product in row.products | 
                 MERGE (com:Compound {smiles: product.smiles})
                 ON CREATE SET com.chemicalName = product.chemical_name, com.appearance = product.appearance,
                               com.inchi = product.inchi, com.molarMass = product.molwt, 
                               com.functionalGroups = product.functional_groups
                 MERGE (rxn)-[:produces]->(com)
                )   
        """
        return reaction_query


if __name__ == "__main__":
    params = dict(
        patents_directory='/home/user/Desktop/5104873',
        number_of_cups=1,
        convert_xml_to_csv=False,
        clean_checker_files=False,
        insert_compounds_with_functional_groups=False,
        insert_change_in_functional_groups=False,
        loading_bars=True,
    )

    graph = Graph()
    InitPatentsIntoNeo4j(**params)

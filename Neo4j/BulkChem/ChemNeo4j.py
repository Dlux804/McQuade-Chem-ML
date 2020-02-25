from time import clock
from math import ceil
from py2neo import Graph, Relationship, NodeMatcher
from rdkit import Chem, DataStructs
import pandas as pd

pd.options.mode.chained_assignment = None  # Help speed up code and drop weird warning pandas can give


class create_relationships:  # Class to generate the different relationship protocols

    """
    Various functions in this class use lambada functions inside of a pandas DataFrame. Basically what a lambada
    function does inside of the DataFrames is it a  function is run for each value inside of a column. So if we had a
    column: list(df['test']) = ['a', 'b', 'c', 'd']; then the function will be run for a, b, c, and d as separate
    variable inputs. Essentially the lambada function acts as a for loop, but is far more efficient since pandas is able
    to implement C code that makes the code run much faster. It is still going to be inefficient to compare every
    molecule to every other molecule no matter what, but this helps to speed things up significantly.
    """

    """
    Protocol Rdkit_sim_score is a relationship protocol that will relate EVERY molecule to EVERY OTHER molecule
    with a relationship labeled 'rdkit_sim_score'.

    Each rdkit_sim_score relationship has an attribute with the score
    that is given when the similarity test is run comparing two molecules. This can be useful because it takes time to
    calculate the rdkit similarity score, and quickly telling what molecules are similar to other ones can be useful.
    There is an fundamental issue with this, however, which is that there are a lot of operations that need to be
    writen. Since there is a relationship relating every molecule to every other molecule by definition, and there
    are ~3000 molecules in the bulkchem data, that is ~3000+2999+2998+...+1 relationships
    (The general formula for this is n(n+1)/2, meaning there are ~4.5 million relationships). Efficiency is needed,
    this file attempts to increase the efficiency of inserting bulk relationships

    The relationship protocol uses the staticmethod 'compare_rdkit_score' which actually compares the rdkit scores
    of one molecule to another.
    """

    @staticmethod
    def compare_rdkit_score(testing_smiles, current_mol):  # Compare rdkit score
        testing_mol = Chem.MolFromSmiles(testing_smiles)
        testing_fingerprint = Chem.RDKFingerprint(testing_mol)
        current_fingerprint = Chem.RDKFingerprint(current_mol)
        sim_score = DataStructs.FingerprintSimilarity(testing_fingerprint, current_fingerprint)
        return sim_score

    def protocol_Rdkit_sim_score(self, node, node_dict):  # Relate only using rdkit score
        self.tx = self.graph.begin()
        self.testing_df['rdkit_sim_score'] = self.testing_df['canonical_smiles'].map(
            lambda x: self.compare_rdkit_score(x, Chem.MolFromSmiles(node_dict['canonical_smiles'])))
        self.testing_df['canonical_smiles'].map(
            lambda x: self.bulk_to_bulk(self.get_prop(x, 'Node'), node, 'Rdkit_Sim_Score',
                                        rdkit_sim_score=self.get_prop(x, 'rdkit_sim_score')))
        self.tx.commit()  # Merge all the relationships created

    """
    This relationship protocol compares pkas of different molecules to one another.
    
    The predicted pkas for each smiles is located in the BulkChemData csv. The pkas were predicted by first
    determining the different functional groups each molecules has, and predicting the pka of each of the functional
    groups. Each functional group in a given molecule will have a different will protonate at different pHs, thus each
    functional group site will have its own pka value. That is the foundation for this comparison protocol, and if a
    molecule has a relating protonation site (or basically a matching functional group in retrospect) it is related.
    
    This relationship uses the 'compare_pka' to compare pkas, which scans the list string in bulkchem under the 
    column 'pka' for any matching pka values. 
    """

    def compare_pka(self, testing_pka, testing_node, current_pka, current_node):
        if str(current_pka) == 'nan':
            return
        if str(testing_pka) == 'nan':
            return
        current_pka = current_pka.split(',')
        testing_pka = testing_pka.split(',')
        for pka_site in testing_pka:
            if pka_site in current_pka:
                self.bulk_to_bulk(testing_node, current_node, 'similar_pka', pka_site=pka_site)

    def protocol_pka(self, node, node_dict):
        self.tx = self.graph.begin()  # Start bulk relationship transaction object to save relationships to
        current_pka = node_dict['pka']
        self.testing_df['canonical_smiles'].map(
            lambda x: self.compare_pka(self.get_prop(x, 'pka'), self.get_prop(x, 'Node'),
                                       current_pka, node))
        self.tx.commit()  # Commit relationships

    """
    """

    def compare_fragments(self, testing_fragments, testing_node, current_fragments, current_node):
        testing_fragments = str(testing_fragments)
        if testing_fragments != 'nan':
            testing_fragments = testing_fragments.split(", ")
            if testing_fragments == current_fragments:
                self.bulk_to_bulk(current_node, testing_node, 'same_fragments', fragments=testing_fragments)

    def protocol_strict_fragments(self, node, node_dict):
        self.tx = self.graph.begin()
        current_fragments = str(node_dict['fragments'])
        if current_fragments != "nan":
            current_fragments = current_fragments.split(", ")
            self.testing_df['canonical_smiles'].map(
                lambda x: self.compare_fragments(self.get_prop(x, 'fragments'), self.get_prop(x, 'Node'),
                                                 current_fragments, node)
            )
        self.tx.commit()

    """
    All the relationship protocols rely on these function. This is to mass create relationships that can be inserted
    in bulk, and not one by one. 
    
    'get_prop' will take a row from the working DataFrame with canonical_smiles matching
    to the con smiles passed to the function. Then it was convert the row into a dictionary that can be queried. Then
    the property in question can be pulled from the dict. This is likely not the most efficient way of doing this and is
    to be improved later on. 
    
    'bulk_to_bulk' will relate to bulk_chemical molecules to one another. Both nodes that are to be inserted must be
    passed, as well as the relationship relating them. Then properties for the relationship can also be passed. For
    example, say we wanted to relate two molecules by matching functional groups. We can do this in a number of ways,
    but lets assume we want to have a relationship called 'has_matching_function_group' with a properties stating 
    the name of the functional group that is relating them (assume function group is amine). We would insert into this 
    function: bulk_to_bulk(node_1, node_2, 'has_matching_function_group', functional_group='amine'). Now this saves
    in memory the relationship, but does not commit the relationship to the graph database. It is far too slow to commit
    each relationship one by one to Neo4j. So multiple relationship can be loaded into memory at once, then batch
    committed later on.
        -Batch commits can be done in bulk as long as the transaction object (.tx object) is a class variable that
         be interacted with and modified by multiple functions. 
    """

    def get_prop(self, con_smiles, prop):  # TODO look for a better way to get properties from DataFrame
        prop_dict = self.testing_df.loc[self.testing_df['canonical_smiles'] == con_smiles].to_dict('records')[0]
        return prop_dict[prop]

    def bulk_to_bulk(self, testing_node, current_node, relationship, **properties):
        Rel = Relationship(testing_node, relationship, current_node, **properties)
        self.tx.create(Rel)  # Create individual relationship, but do not commit to neo4j

    """
    These methods are the backbone methods that are needed for the main function to work properly.
    
    The timer keeps track of the time needed until all entities are related and inserted. Instead of explaining how
    the math works in words in this file, please refer to 'bulkchem_datafiles/Time_vs_molecules_demo.xlsx'. The file
    can serve as a visual explanation to understand the math behind this function. But in summary, the time needed to 
    trained is directly related to number of operations remaining. Time Remaining = m * Operations left:
    where m = change in time/change in operations. The m is a changing average that is calculated as the relationships
    are inserted. This function basically calculates the moving average m, then multiples that by the number of predict
    operations left.
        Now, the actual number of operations will vary in reality. Since it is not possible to know exactly how many 
        total operations will be needed, as we do not know how many relationships will be inserted into the GB. So 
        this timer only acts as an estimate, and will not have perfect predicating power. But, it is assumed that
        all molecules will have a relationship, so the predict acts as a upper limit for the max amount of time the
        script will need to run based on the moving average.
    
    The '__get_testing_df__' function is responsible for loading in a set number of nodes at a time. For bulkchem
    database, there are a small number of molecules (nodes) and all of them can be loaded into memory safely. But, for
    larger databases, such as PubChem, this is not a good idea. So this functions will split the nodes in the Neo4j
    GB into chunks that can be digested and inserted in bulk safely. So if there were 300 total nodes, and each node
    was to be compared with each other node and the max nodes to keep in memory is 100, then there would be a max of 3
    chunks to digest. I say max because if more than 100 nodes have already been compared, then there are only
    200 nodes left to compare. And since there 100 max nodes allowed to be keep in memory, this difference has to be
    keep track of. This difference is easiest to be keep track of in the main function. 
        So this function keeps track of the number of node remaining and the max number of nodes to keep in
        memory, and return the testing DataFrame accordingly. 
    """

    def timer(self, time_for_batch):  # Keep track of long it will take to finish
        if not self.o:  # Please refer to excel spreadsheet to understand how this function was derived
            n = self.len_nodes
            self.o = n * (n + 1) / 2
        ni = self.molecules_remaining
        oi = ni * (ni + 1) / 2
        delta_o = self.o - oi
        m = (time_for_batch / delta_o)
        if not self.m:
            self.m = m
        self.m = ((self.counter - 1) * self.m + m) / self.counter
        self.o = oi
        time_needed = self.m * self.o
        return time_needed

    def __get_testing_df__(self, i):  # Get DataFrame to compare molecule to
        lower_limit = i * self.max_nodes_in_ram + self.counter
        upper_limit = lower_limit + self.max_nodes_in_ram
        bulk_dicts = []
        counter = lower_limit
        for node in self.raw_nodes.skip(lower_limit):
            if counter == upper_limit:
                break
            molecule = (dict(node))
            molecule['Node'] = node
            bulk_dicts.append(molecule)
            counter = counter + 1
        return pd.DataFrame(bulk_dicts)

    """
    This is the main function that wraps all the functions in this class. 
    
    The idea is to compare each node to each other node, and create a relationship based on a set of rules. 
    There are many ways to compare each node to each other node, but this scripts aims at efficiency. So the idea
    to take a single molecule and declare it as the current_node, and compare this to the rest of the nodes, which we
    will call testing nodes. Then take all the testing nodes, and shove them into a DataFrame (or DataFrame chunks if
    dataset is large enough), and compare all the nodes all at once in batch, then insert the found relationships in
    batch. To get a better idea of all the different moving parts, please refer to the different functions found above.
    Besides being the brain of this script, this function is also responsible for using the timer function and printing
    the time remaining, as well as making a csv file containing the time_vs_molecule(remaining) to further inspection
    of the timer.
    """

    def __main__(self):  # Main loop, compare all molecules to each other
        for node in self.raw_nodes:
            time_for_batch = clock()
            self.counter = self.counter + 1
            self.molecules_remaining = self.len_nodes - self.counter
            print("{0} molecules left to compare".format(self.molecules_remaining))  # Let user know amount left
            node_dict = (dict(node))
            for i in range(self.splits):
                if (i * self.max_nodes_in_ram + self.max_nodes_in_ram) <= self.counter:
                    pass
                elif i == self.splits - 1 and self.counter >= self.abs_upper_limit:
                    pass
                else:
                    self.testing_df = self.__get_testing_df__(i)
                    self.protocol_dict[self.protocol](node, node_dict)
            time_for_batch = clock() - time_for_batch
            time_left_minutes = round(self.timer(time_for_batch) / 60, 2)
            time_left_hours = round(time_left_minutes / 60, 2)
            self.time_df = self.time_df.append({'Molecules Remaining': self.molecules_remaining,
                                                'Time needed (s)': time_for_batch,
                                                'Total Time passed (min)': self.run_time,
                                                'Predicted Time Left (min)': time_left_minutes}, ignore_index=True)
            print("\nTime Remaining: {0} minutes ({1} hours)".format(time_left_minutes, time_left_hours))
            self.time_df.to_csv('bulkchem_datafiles/Time_vs_molecules.csv', index=False)

    def __init__(self, protocol, max_nodes_in_ram=3000):

        self.run_time = clock()  # Declare variables for timer
        self.average_time = None
        self.o = None
        self.m = None
        self.time_df = pd.DataFrame(columns=['Molecules Remaining', 'Time needed (s)', 'Total Time passed (min)',
                                             'Predicted Time Left (min)'])

        # Please note the timer is still under development and testing, time may not be accurate

        self.graph = Graph()  # Get neo4j graph database # TODO allow user to enter credentials to use remote GBs
        self.tx = None  # Prepare the transaction object that will hold all the relationships that will be bulk inserted

        self.protocol = protocol  # Define protocols
        self.protocol_dict = {
            "Rdkit_sim_score": self.protocol_Rdkit_sim_score,
            "pka": self.protocol_pka,
            "strict_fragments": self.protocol_strict_fragments
        }

        if self.protocol not in list(self.protocol_dict.keys()):  # Make sure protocol user gave exists
            raise Exception('Protocol not found')

        matcher = NodeMatcher(self.graph)  # Get nodes from neo4j database (Doesn't load them in ram yet)
        self.raw_nodes = matcher.match("bulkChemMolecule")

        self.len_nodes = len(self.raw_nodes)

        if self.len_nodes <= 0:
            raise Exception("There are no nodes in the database, consider using init_neo_bulkchem found in backends")

        if max_nodes_in_ram > self.len_nodes:  # Verify max nodes does not exceed number of nodes in database
            self.max_nodes_in_ram = self.len_nodes
        else:
            self.max_nodes_in_ram = max_nodes_in_ram

        self.splits = ceil(self.len_nodes / self.max_nodes_in_ram)  # Split data in neo4j database into chunks
        self.abs_upper_limit = len(self.raw_nodes) - (self.splits - 1) * self.max_nodes_in_ram
        self.counter = 0

        print('Comparing Bulk Chem Data with rule set "{0}"'.format(self.protocol))  # Start comparing
        self.__main__()

from time import clock
from math import ceil
from py2neo import Node, Graph, Relationship, NodeMatcher
from rdkit import Chem, DataStructs
import pandas as pd
from .protocols import FragList_LogP

pd.options.mode.chained_assignment = None  # Help speed up code and drop weird warning pandas can give


class create_relationships:  # Class to generate the different relationship protocols

    def __get_prop__(self, con_smiles, prop):  # TODO look for a better way to get properties from DataFrame
        prop_dict = self.testing_df.loc[self.testing_df['canonical_smiles'] == con_smiles].to_dict('records')[0]
        return prop_dict[prop]

    def __bulk_to_bulk__(self, testing_node, current_node, relationship, **properties):
        Rel = Relationship(testing_node, relationship, current_node, **properties)
        self.tx.create(Rel)  # Create individual relationship, but do not commit to neo4j

    def __init__(self, protocol, max_nodes_in_ram, *files):

        # Please note the timer is still under development and testing, time may not be accurate

        self.protocol = protocol  # Define protocols
        '''
        self.protocol_dict = {
            "Rdkit_sim_score": self.protocol_Rdkit_sim_score,
            "pka": self.protocol_pka,
            "strict_fragments": self.protocol_strict_fragments,
            "strict_rdkit_sim_score": self.protocol_strict_rdkit_sim_score,
            "fragment_list_comparision": fragment_list_comparision
        }
        '''
        self.protocol_dict = {
            "FragList_LogP": FragList_LogP
        }

        if self.protocol not in list(self.protocol_dict.keys()):  # Make sure protocol user gave exists
            raise Exception('Protocol not found')


        print('Comparing Bulk Chem Data with rule set "{0}"'.format(self.protocol))  # Start comparing

        self.protocol_dict[self.protocol].__compare__(max_nodes_in_ram, *files)
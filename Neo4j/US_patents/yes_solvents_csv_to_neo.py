import pandas as pd
import ast
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import Descriptors
import os
from py2neo import Node, Relationship, NodeMatcher, Graph


class file_to_neo4j:

    def __init__(self, file, ignore_checker_files=False):

        self.delete_checker_files = ignore_checker_files
        self.graph = Graph()

        self.insert_dict = {
            "reaction_smiles": self.insert_reaction_smiles,
            "sources": self.insert_sources,
            "reactants": self.insert_reactants,
            "products": self.insert_products,
            "spectators": self.insert_spectators,
            "stages": self.insert_stages
        }

        self.file = file
        if not ignore_checker_files:
            if os.path.exists(self.file + ".checker"):
                return

        self.raw_data = pd.read_csv(file)
        self.tx = self.graph.begin()

        for index, row in self.raw_data.iterrows():
            self.reaction_props = {}
            self.main_reaction_nodes = {}
            self.row = row
            self.row_dict = dict(row)
            self.matcher = NodeMatcher(self.graph)

            for item in self.raw_data:
                self.node_type = item
                self.items = ast.literal_eval(self.row_dict[item])
                self.insert_dict[self.node_type]()

            self.reaction_props['label'] = 'Reaction'
            reaction_node = Node('reaction', **self.reaction_props)
            reaction_node.__primarylabel__ = 'reaction'
            reaction_node.__primarykey__ = 'reaction_smiles'
            for reactant in self.main_reaction_nodes['reactants']:
                reacts = Relationship.type('reacts')
                self.tx.merge(reacts(reactant, reaction_node))
            for product in self.main_reaction_nodes['products']:
                produces = Relationship.type('produces')
                self.tx.merge(produces(reaction_node, product))
            for solvent in self.main_reaction_nodes['solvents']:
                dissolves = Relationship.type('spectates')
                self.tx.merge(dissolves(solvent, reaction_node))
        self.tx.commit()
        open(self.file + ".checker", "a").close()

    @staticmethod
    def list_to_steps_string(items):
        counter = 1
        main_string = []
        for item in items:
            if not main_string:
                main_string = "{}:".format(str(counter)) + item
            else:
                main_string = main_string + " ||| {}:".format(str(counter)) + item
            counter = counter + 1
        return main_string

    def insert_reaction_smiles(self):
        self.reaction_props['reaction_smiles'] = self.list_to_steps_string(self.items)

    def insert_sources(self):
        self.reaction_props['insert_sources'] = self.list_to_steps_string(self.items)

    def insert_stages(self):
        self.reaction_props['insert_stages'] = " ||| ".join(self.items)

    @staticmethod
    def get_molecule_primary_key(prop_dict):
        if 'smiles' in prop_dict.keys():
            return 'smiles'
        elif 'chemical_names' in prop_dict.keys():
            return 'chemical_names'
        else:
            return None

    def get_molecule_props_dict(self, molecule_dict):
        prop_dict = {}
        for prop in molecule_dict:
            for value in molecule_dict[prop]:
                testing_value = value.split(' = ')
                if len(testing_value) > 1:
                    if testing_value[0] in prop_dict.keys():
                        if isinstance(prop_dict[testing_value[0]], list):
                            prop_dict[testing_value[0]].append(testing_value[1])
                        else:
                            prop_dict[testing_value[0]] = [prop_dict[testing_value[0]], testing_value[1]]
                    else:
                        prop_dict[testing_value[0]] = testing_value[1]
                else:
                    if prop in prop_dict.keys():
                        if prop_dict[prop] == 'title compound':
                            prop_dict[prop] = value
                        elif isinstance(prop_dict[prop], list):
                            prop_dict[prop].append(value)
                        else:
                            prop_dict[prop] = [prop_dict[prop], value]
                    else:
                        prop_dict[prop] = value
        try:
            mol = MolFromSmiles(str(prop_dict['smiles']))
            con_smiles = MolToSmiles(mol)
            prop_dict['smiles'] = con_smiles
            prop_dict['molar_mass'] = Descriptors.ExactMolWt(mol)
        except TypeError:
            prop_dict = None
        except KeyError:
            prop_dict = None
        return prop_dict

    def insert_and_retrieve_compound_node(self, molecule_dict):
        prop_dict = self.get_molecule_props_dict(molecule_dict)
        if prop_dict is None:
            return None
        compound_node = Node('compound', **prop_dict)
        compound_node.__primarylabel__ = 'compound'
        compound_node.__primarykey__ = self.get_molecule_primary_key(prop_dict)
        return compound_node

    def insert_reactants(self):
        self.main_reaction_nodes['reactants'] = []
        for molecule_dict in self.items:
            compound_node = self.insert_and_retrieve_compound_node(molecule_dict)
            if compound_node is not None:
                self.main_reaction_nodes['reactants'].append(compound_node)

    def insert_products(self):
        self.main_reaction_nodes['products'] = []
        for molecule_dict in self.items:
            compound_node = self.insert_and_retrieve_compound_node(molecule_dict)
            if compound_node is not None:
                self.main_reaction_nodes['products'].append(compound_node)

    def insert_spectators(self):
        self.main_reaction_nodes['solvents'] = []
        for molecule_dict in self.items:
            compound_node = self.insert_and_retrieve_compound_node(molecule_dict)
            if compound_node is not None:
                self.main_reaction_nodes['solvents'].append(compound_node)
import ast

from py2neo import Node, Relationship, NodeMatcher, Graph


class data_to_neo4j:

    def __init__(self, input_data):

        self.input_data = input_data
        self.graph = Graph()
        self.matcher = NodeMatcher(self.graph)

        self.insert_dict = {
            "reaction_smiles": self.insert_reaction_smiles,
            "sources": self.insert_sources,
            "stages": self.insert_stages
        }

        self.tx = self.graph.begin()

        for index, row in self.input_data.iterrows():
            self.reaction_props = {}
            self.row = row
            self.row_dict = dict(row)
            self.matcher = NodeMatcher(self.graph)

            for item in self.input_data:
                if item in self.insert_dict.keys():
                    self.node_type = item
                    self.items = ast.literal_eval(self.row_dict[item])
                    self.insert_dict[self.node_type]()

            self.reaction_props['label'] = 'Reaction'
            reaction_node = Node('reaction', **self.reaction_props)
            reaction_node.__primarylabel__ = 'reaction'
            reaction_node.__primarykey__ = 'reaction_smiles'
            self.tx.create(reaction_node)

            for reactant in self.row_dict['reactants']:
                reactant = self.fetch_node(reactant)
                rel = Relationship(reactant, 'reacts', reaction_node)
                self.tx.create(rel)
            for product in self.row_dict['products']:
                product = self.fetch_node(product)
                rel = Relationship(reaction_node, 'produces', product)
                self.tx.create(rel)
            for solvent in self.row_dict['solvents']:
                solvent = self.fetch_node(solvent)
                rel = Relationship(solvent, 'solvent', reaction_node)
                self.tx.create(rel)
            for catalyst in self.row_dict['catalyst']:
                catalyst = self.fetch_node(catalyst)
                rel = Relationship(catalyst, 'catalyzes', reaction_node)
                self.tx.create(rel)
        self.tx.commit()

    def fetch_node(self, node_str):
        try:
            try:
                node_ID = int(node_str)
                return self.matcher.get(node_ID)
            except ValueError:
                node_ID = node_str.split(':')[0]
                node_ID = node_ID[2:len(node_ID)]
                return self.matcher.get(int(node_ID))
            except TypeError:
                node_ID = node_str.split(':')[0]
                node_ID = node_ID[2:len(node_ID)]
                return self.matcher.get(int(node_ID))
        except AttributeError:
            return node_str

    @staticmethod
    def test_compound(compound):
        testing_str = str(compound)
        testing_str = testing_str.strip()
        testing_str = testing_str.strip('[]')
        if testing_str == '':
            return None
        return compound

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

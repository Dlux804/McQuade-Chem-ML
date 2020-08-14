"""
Objective: Batch submit time and memory intensive imports using APOC.
"""
from py2neo import ClientError
# TODO Add docstring


class Query:

    def __init__(self, graph):
        self.graph = graph  # Connecting python to Neo4j using py2neo

    def __molecules_and_rdkit2d_query__(self):
        """
        Objective: Cypher query that creates nodes and relationships between SMILES and rdkit2d features
        :return:
        """
        restraint_strings = ["""
                    CREATE CONSTRAINT ON 
                    (n:Feature) ASSERT n.name IS UNIQUE
                    """,
                             """
                CREATE CONSTRAINT ON 
                (n:FeatureMethod) ASSERT n.feature IS UNIQUE
                """]

        for restraint_string in restraint_strings:
            try:
                tx = self.graph.begin(autocommit=True)
                tx.evaluate(restraint_string)
            except ClientError:
                pass
        mol_rdkit2d_query = """
                MERGE (rdkit2d:FeatureMethod {feature:'rdkit2d'})
                WITH rdkit2d
                UNWIND $molecules as molecule
                    MERGE (mol:Molecule {SMILES: molecule.smiles})
                    FOREACH (feat in molecule.feats | 
                        MERGE (feature:Feature {name: feat.name})
                        MERGE (mol)-[:HAS_DESCRIPTOR {value:feat.value, feat_name:feat.name}]->(feature)
                        MERGE (feature)<-[r:CALCULATES]-(rdkit2d)
                            )
                    """
        return mol_rdkit2d_query

    def __make_molecules_query__(self, target_name):
        """
        Objective: Cypher Query to make molecule nodes and their properties
        :param target_name: True name of target column. The name is stored in core/storage/dictionaries.py
        :return:
        """
        restraint_string = """
                    CREATE CONSTRAINT ON 
                    (n:Molecule) ASSERT n.SMILES IS UNIQUE
                """

        try:
            tx = self.graph.begin(autocommit=True)
            tx.evaluate(restraint_string)
        except ClientError:
            pass
        molecule_query = """
            UNWIND $molecules as molecule
            MERGE (mol:Molecule {SMILES: molecule.smiles})
                ON CREATE Set mol.dataset = [$dataset], mol.%s = molecule.target, mol.name = "Molecule"
            """ % target_name

        return molecule_query

    def __fragment_query__(self):
        """
        Objective: Cypher query that creates nodes for fragments
        :return:
        """
        restraint_string = """
                CREATE CONSTRAINT ON 
                (n:Fragments) ASSERT n.name IS UNIQUE
            """

        try:
            tx = self.graph.begin(autocommit=True)
            tx.evaluate(restraint_string)
        except ClientError:
            pass

        fragment_query = """
            CALL apoc.periodic.iterate(
                    "
                    UNWIND $rows as row
                    RETURN row
                    ",
                    "
                    MERGE (mol:Molecule {SMILES: row.smiles})
                        FOREACH (fragment in row.fragments |
                            MERGE (frag:Fragments {name: fragment})
                            MERGE (mol)-[:HAS_FRAGMENTS]->(frag)
                            )
                    ",
                    {batchSize:2000, parallel:True, params:{rows:$rows}})
                    """
        return fragment_query

    def __check_for_constraints__(self):
        """
        This function will check and make sure that constraints are on the graph. This is important because
        constraints allow nodes to replace IDs with, well the constraint label. This allows Neo4j to quickly merge
        the different nodes. Constraints plus UNWIND allow for some really impressive speed up times in inserting
        data into Neo4j.
        """

        constraint_check_strings = [
            """
        CREATE CONSTRAINT unique_MLModel_name
        ON (n:MLModel)
        ASSERT n.name IS UNIQUE
        """,

            """
        CREATE CONSTRAINT unique_FeatureList_feat_ID
        ON (n:FeatureList)
        ASSERT n.feat_ID IS UNIQUE
        """,

            """
        CREATE CONSTRAINT unique_Dataset_name
        ON (n:Dataset)
        ASSERT n.data IS UNIQUE
        """,
            """
        CREATE CONSTRAINT unique_Algorithm_name
        ON (n:Algorithm)
        ASSERT n.name IS UNIQUE
            """,
            """
        CREATE CONSTRAINT unique_TestSet_run_name
        ON (n:TestSet)
        ASSERT n.run_name IS UNIQUE
        
            """,
            """
        CREATE CONSTRAINT unique_TrainSet_run_name
        ON (n:TrainSet)
        ASSERT n.run_name IS UNIQUE
        
            """,
            """
        CREATE CONSTRAINT unique_ValSet_run_name
        ON (n:ValSet)
        ASSERT n.run_name IS UNIQUE
        
            """

        ]

        for constraint_check_string in constraint_check_strings:
            try:
                tx = self.graph.begin(autocommit=True)
                tx.evaluate(constraint_check_string)
            except ClientError:
                pass

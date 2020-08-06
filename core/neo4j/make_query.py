"""
Objective: Batch submit time and memory intensive imports using APOC.
"""
from py2neo import ClientError


class Query:
    def __init__(self, size):
        self.size = size
        # if self.size >= 10000:
        #     print("This dataset has more than 10000 SMILES. Switching to apoc for batch import")
        #     print("This is gonna take a while. Might as well grab some food while you wait for it")
        # else:
        #     print("No batch import needed. Full steam ahead!!!!")

    def __molecules_and_rdkit2d_query__(self):
        """"""
        if self.size >= 10000:

            mol_rdkit2d_query = """
           UNWIND $molecule as molecule
           With molecule
           CALL apoc.periodic.iterate('
           MATCH (mol:Molecule {SMILES: $mols.smiles})
           MERGE (rdkit2d:FeatureMethod {feature:"rdkit2d"})

               FOREACH (feat in $mols.feats|
                   MERGE (feature:Feature {name: feat.name})
                   MERGE (mol)-[:HAS_DESCRIPTOR {value: feat.value, feat_name:feat.name}]->(feature)
                   MERGE (feature)<-[r:CALCULATES]-(rdkit2d)
                       )
           ',';',
               {batchSize:100000, parallel:true, params:{mols:molecule}}) YIELD batches, total
           RETURN batches, total
           """
        else:
            mol_rdkit2d_query = """
            UNWIND $molecule as molecule
            MATCH (mol:Molecule {SMILES: molecule.smiles})
            MERGE (rdkit2d:FeatureMethod {feature:"rdkit2d"})
            FOREACH (feat in molecule.feats|
                MERGE (feature:Feature {name: feat.name})
                MERGE (mol)-[:HAS_DESCRIPTOR {value: feat.value, feat_name:feat.name}]->(feature)
                MERGE (feature)<-[r:CALCULATES]-(rdkit2d)
                    )

        """
        return mol_rdkit2d_query

    def __make_molecules_query__(self):
        """"""
        if self.size >= 10000:
            molecule_query = """
        UNWIND $molecules as molecule
        With molecule as molecule, $dataset as dataset, $target_name as target_name
        CALL apoc.periodic.iterate('
        MERGE (mol:Molecule {SMILES: $mols.smiles, name: "Molecule"})
        Set mol.dataset = [$data], mol.target = [$mols.target]
        ',';',
        {batchSize:100000, parallel:true, params:{mols:molecule, data:dataset, exp:target_name}}) YIELD batches, total
            RETURN batches, total
                """
        else:
            molecule_query = """
                    UNWIND $molecules as molecule
                    MERGE (mol:Molecule {SMILES: molecule.smiles, name: "Molecule"})
                    Set mol.dataset = [$dataset], mol.target = [molecule.target], mol.target_name = [$target_name]
                    """
        return molecule_query

    def __fragment_query__(self):
        """"""
        if self.size >= 10000:
            fragment_query = """
                UNWIND $fragments as fragment
                With fragment
                CALL apoc.periodic.iterate('
                MATCH (mol:Molecule {SMILES: $frags.smiles})
                    FOREACH (value in $frags.fragments|
                        MERGE (fragment:Fragments {name: value.fragments})
                        MERGE (mol)-[:HAS_FRAGMENTS]->(fragment)
                            )
                            ',';',
                {batchSize:100000, parallel:true, params:{frags:fragment}}) YIELD batches, total
            RETURN batches, total
                """

        else:
            fragment_query = """
            UNWIND $fragments as fragment
            MATCH (mol:Molecule {SMILES: fragment.smiles})
            FOREACH (value in fragment.fragments|
                    MERGE (fragment:Fragments {name: value.fragments})
                    MERGE (mol)-[:HAS_FRAGMENTS]->(fragment)
                    )
                    """
        return fragment_query

    def __check_for_constraints__(self, g):
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
        CREATE CONSTRAINT unique_Feature_name
        ON (n:Feature)
        ASSERT n.name IS UNIQUE
        """,

            """
        CREATE CONSTRAINT unique_Molecule_SMILES
        ON (n:Molecule)
        ASSERT n.SMILES IS UNIQUE
        """,

            """
            CREATE CONSTRAINT unique_Fragments_name
            ON (n:Fragments)
            ASSERT n.name IS UNIQUE
            """

        ]

        for constraint_check_string in constraint_check_strings:
            try:
                tx = g.begin(autocommit=True)
                tx.evaluate(constraint_check_string)
            except ClientError:
                pass

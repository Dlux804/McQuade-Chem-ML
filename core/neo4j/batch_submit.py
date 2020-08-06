"""
Objective: Batch submit time and memory intensive imports using APOC.
"""


def __molecules_and_rdkit2d_query__(size):
    """"""
    if size >= 10000:
        print("This dataset has more than 10000 SMILES. Switching to apoc for batch import\\")
        print("This is gonna take a while. Might as well grab some food while you wait for it")
        mol_rdkit2d_query = """
       UNWIND $molecule as molecule
       With molecule
       CALL apoc.periodic.iterate('
       Match (mol:Molecule {SMILES: $mols.smiles, name: "Molecule"})
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
        print("No batch import needed. Full steam ahead!!!!")
        mol_rdkit2d_query = """
        UNWIND $molecule as molecule
        MERGE (rdkit2d:FeatureMethod {feature:"rdkit2d"})
        MERGE (mol:Molecule {SMILES: molecule.smiles})
        FOREACH (feat in molecule.feats|
            MERGE (feature:Feature {name: feat.name})
            MERGE (mol)-[:HAS_DESCRIPTOR {value: feat.value, feat_name:feat.name}]->(feature)
            MERGE (feature)<-[r:CALCULATES]-(rdkit2d)
                )
    
    """
    return mol_rdkit2d_query


def __make_molecules_query__(size):
    """"""
    if size >= 10000:
        print("This dataset has more than 10000 SMILES!!! Switching to apoc for batch import\\")
        print("This is gonna take a while. Might as well grab some food while you're at it")
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
        print("No batch import needed. Full steam ahead!!!!")
        molecule_query = """
                UNWIND $molecules as molecule
                MERGE (mol:Molecule {SMILES: molecule.smiles, name: "Molecule"})
                Set mol.dataset = [$dataset], mol.target = [molecule.target], mol.target_name = [$target_name]
                """
    return molecule_query


def __fragment_query__(size):
    """"""
    if size >= 10000:
        fragment_query = """
            UNWIND $fragments as fragment
            With fragment
            CALL apoc.periodic.iterate('
            MERGE (mol:Molecule {SMILES: $frags.smiles})
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
        MERGE (mol:Molecule {SMILES: fragment.smiles})
        FOREACH (value in fragment.fragments|
                MERGE (fragment:Fragments {name: value.fragments})
                MERGE (mol)-[:HAS_FRAGMENTS]->(fragment)
                )
                """
    return fragment_query

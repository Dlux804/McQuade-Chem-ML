from tqdm import tqdm

import pandas as pd
from py2neo import Graph, ClientError
from rdkit.Chem import RDKFingerprint, MolFromSmiles, DataStructs

from core.storage import calculate_fragments
from core.features import canonical_smiles


class MoleculeSimilarity:

    """
    Goal is to be able to run various similarity algorithms to determine how similar one molecule is
    to every other molecule in a Neo4j database.
    """

    def __init__(self, graph):
        self.graph = graph

    def hyer_sim(self, smiles, limit=None):

        """
        Crude method of trying to compare how similar two molecules are. 0 is least similar 1 is completely similar.
        This method relies strictly on how many fragments each molecule has in common.
        formula: 2*|A|U|B| / (|A| + |B|)

        All the math for this comparison is done in python

        :param smiles: Control smiles to run similarity scores against
        :param limit: Return top nth number of results
        :return: Dataframe of smiles and similarity score
        """

        query_results = self.graph.run("""
        
            MATCH (n:Molecule {smiles: $smiles})
    
            MATCH (m:Molecule)
            WHERE m.smiles <> n.smiles
            
            MATCH (n)-[:HAS_FRAGMENT]->(frag:Fragment)
            WITH collect(frag.name) as fragments_1, n, m
            
            MATCH (m)-[:HAS_FRAGMENT]->(frag:Fragment)
            WITH collect(frag.name) as fragments_2, fragments_1, n, m
            
            MATCH (n)-[:HAS_FRAGMENT]->(frag:Fragment)<-[:HAS_FRAGMENT]-(m)
            WITH collect(frag.name) as fragments_both, fragments_1, fragments_2, m.smiles as smiles
            
            RETURN smiles, fragments_1, fragments_2, fragments_both
        
        """, parameters={'smiles': smiles}).data()

        sim_results = []
        for row in query_results:
            both = len(row['fragments_both'])
            total = len(row['fragments_1']) + len(row['fragments_2'])
            row['similarity'] = 2 * both / total
            sim_results.append(row)
        sim_results = pd.DataFrame(sim_results)
        sim_results = sim_results[['smiles', 'similarity']]
        sim_results = sim_results.sort_values(by=['similarity'], ascending=False)
        sim_results = sim_results.reset_index(drop=True)
        if limit:
            sim_results = sim_results.head(limit)
        return sim_results

    def jaccard_sim(self, smiles, limit=None):

        """
        Crude method of trying to compare how similar two molecules are. 0 is least similar 1 is completely similar.
        This method relies strictly on how many fragments each molecule has in common.
        formula: |A|U|B| / (|A| + |B| - |A|U|B|)

        All the math for this comparison is done in Neo4j

        :param smiles: Control smiles to run similarity scores against
        :param limit: Return top nth number of results
        :return: Dataframe of smiles and similarity score
        """

        query_results = self.graph.run("""
        
            MATCH (mol_1:Molecule)-[:HAS_FRAGMENT]->(frags_1:Fragment)
                WHERE mol_1.smiles = $smiles
                WITH mol_1, collect(id(frags_1)) as frags_1
            
            MATCH (mol_2:Molecule)-[:HAS_FRAGMENT]->(frags_2:Fragment)
                WHERE mol_2.smiles <> mol_1.smiles
                WITH mol_1, frags_1, mol_2, collect(id(frags_2)) as frags_2
            
            RETURN mol_2.smiles as smiles, 
                gds.alpha.similarity.jaccard(frags_1, frags_2) AS similarity
            
        
        """, parameters={'smiles': smiles}).data()

        sim_results = pd.DataFrame(query_results)
        sim_results = sim_results.sort_values(by=['similarity'], ascending=False)
        sim_results = sim_results.reset_index(drop=True)
        if limit:
            sim_results = sim_results.head(limit)
        return sim_results

    def rdkit_sim(self, smiles, limit=None):

        """
        Refined method of comparing how similiar two molecules are based on their RDKfinerprint.
        Please see the RDkit documentation for further information on this comparison.
        https://www.rdkit.org/docs/source/rdkit.DataStructs.html

        All calculations are made in python, but molecules are grabbed from the Neo4j server first.

        :param smiles: Control smiles to run similarity scores against
        :param limit: Return top nth number of results
        :return: Dataframe of smiles and similarity score
        """

        control_fingerprint = RDKFingerprint(MolFromSmiles(smiles))

        query_results = self.graph.run("""
        
            MATCH (mol:Molecule)
            WHERE mol.smiles <> $smiles
            RETURN mol.smiles as smiles
        
        """, parameters={'smiles': smiles}).data()

        sim_results = []
        for smiles in pd.DataFrame(query_results)['smiles'].tolist():
            testing_fingerprint = RDKFingerprint(MolFromSmiles(smiles))
            sim_score = DataStructs.FingerprintSimilarity(control_fingerprint, testing_fingerprint)
            sim_results.append({'smiles': smiles, 'similarity': sim_score})
        sim_results = pd.DataFrame(sim_results)
        sim_results = sim_results.sort_values(by=['similarity'], ascending=False)
        sim_results = sim_results.reset_index(drop=True)
        if limit:
            sim_results = sim_results.head(limit)
        return sim_results

    def compare_sim_algorithms(self, smiles, file=None, limit=None):
        hyer = self.hyer_sim(smiles=smiles)
        hyer = hyer.rename(columns={"similarity": "hyer_similarity"})

        jaccard = self.jaccard_sim(smiles=smiles)
        jaccard = jaccard.rename(columns={"similarity": "jaccard_similarity"})

        rdkit = self.rdkit_sim(smiles=smiles)
        rdkit = rdkit.rename(columns={"similarity": "rdkit_similarity"})

        sim_results = pd.merge(rdkit, hyer, on="smiles")
        sim_results = pd.merge(sim_results, jaccard, on="smiles")
        sim_results['percent_error_rdkit_hyer'] = (abs(sim_results['rdkit_similarity'] - sim_results['hyer_similarity']) / sim_results['rdkit_similarity']) * 100
        sim_results['percent_error_rdkit_jaccard'] = (abs(sim_results['rdkit_similarity'] - sim_results['jaccard_similarity']) / sim_results['rdkit_similarity']) * 100
        sim_results = round(sim_results, 3)

        if limit:
            sim_results = sim_results.head(limit)
        if file:
            sim_results.to_csv(file)
        return sim_results


def insert_dataset_molecules(graph, df):

    try:
        graph.evaluate("CREATE CONSTRAINT ON (n:Molecule) ASSERT n.smiles IS UNIQUE")
        graph.evaluate("CREATE CONSTRAINT ON (n:Fragment) ASSERT n.name IS UNIQUE")
    except ClientError:
        pass

    print('Calculating Fragments...')
    rows = []
    for row in tqdm(df.to_dict('records')):
        smiles = row['smiles']
        fragments = calculate_fragments(smiles)
        rows.append({'smiles': smiles, 'fragments': fragments})

    print('Inserting molecules with fragments...')
    graph.evaluate(
        """
        UNWIND $rows as row
            MERGE (mol:Molecule {smiles: row['smiles']})
            WITH mol, row
            UNWIND row['fragments'] as fragment
                MERGE (frag:Fragment {name: fragment})
                MERGE (mol)-[:HAS_FRAGMENT]->(frag)
        """, parameters={'rows': rows}
    )


def cleanup_smiles(file):
    df = pd.read_csv(file)
    df = canonical_smiles(df)
    df.to_csv(file, index=False)


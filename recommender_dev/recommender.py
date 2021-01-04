import os
import random
import shutil

import pandas as pd
from py2neo import Graph

from core import MlModel
from recommender_dev.molecules import insert_dataset_molecules, MoleculeSimilarity


def check_for_results_folder(results_directory):
    if not os.path.exists(results_directory):
        os.chdir(results_directory)


def delete_current_results(results_directory):
    for sub_directory in os.listdir(results_directory):
        path = f'{results_directory}/{sub_directory}'
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


class Recommender:

    def __init__(self, smiles):

        """
        Recommender is a development class that attempts to try and recommend a model for the user given a smiles.
        The goal is to find similar molecules, and recommend models that did well for those molecules.

        """

        self.control_smiles = smiles
        self.graph = None
        self.rdkit_sim = None
        self.jaccard_sim = None
        self.hyer_sim = None
        self.compare_sim_results = None
        self.rdkit_results = []
        self.jaccard_results = []
        self.hyer_results = []
        self.control_results = []

    def connect_to_neo4j(self, port="bolt://localhost:7687", username="neo4j", password="password"):
        """

        :param port: Port to connect to Neo4j (can be http or bolt)
        :param username: Username for Neo4j database
        :param password: Password for Neo4j database
        :return:
        """

        self.graph = Graph(port, username=username, password=password)

    def insert_molecules_into_neo4j(self, dataset):
        """
        Inserts molecules in passed dataset into Neo4j

        :param dataset: File to insert smiles
        :return:
        """

        if self.graph is None:
            raise AttributeError(f"Cannot insert molecules into unspecified graph, run self.connect_to_neo4j()")

        query_results = self.graph.run("""
            MATCH (mol:Molecule)
            RETURN mol.smiles
        """).data()

        if len(query_results) == 0:
            df = pd.read_csv(dataset)
            insert_dataset_molecules(self.graph, df)

    def gather_similar_molecules(self, limit=5):
        """
        Runs molecule.py functions to gather the similar molecules based on the different similarity functions

        :param limit: How many to return as a head (limit=5 means only return top 5 similar molecules)
        :return:
        """

        sim = MoleculeSimilarity(self.graph)
        self.rdkit_sim = sim.rdkit_sim(self.control_smiles, limit=limit)
        self.jaccard_sim = sim.jaccard_sim(self.control_smiles, limit=limit)
        self.hyer_sim = sim.hyer_sim(self.control_smiles, limit=limit)
        self.compare_sim_results = sim.compare_sim_algorithms(self.control_smiles)

    def run_models(self, dataset, target, tune=None, cv=None, opt_iter=None,
                   learners=None, features=None):

        if self.rdkit_sim is None:
            raise AttributeError(f"Similar molecules not found, run self.gather_similar_molecules()")

        print('\nGenerating models...')

        if tune is None:
            tune = False
        if cv is None:
            cv = 5
        if opt_iter is None:
            opt_iter = 100
        if learners is None:
            learners = ['rf', 'gdb']
        if features is None:
            features = [[0], [2], [3], [4], [0, 2], [0, 3]]

        print(f'List of learners: {str(learners)}')
        print(f'List of features: {str(features)}')
        print(f'Number of models to run: {len(learners) * len(features)}')

        test_smiles = [self.control_smiles]
        test_smiles.extend(self.rdkit_sim['smiles'].tolist())
        test_smiles.extend(self.jaccard_sim['smiles'].tolist())
        test_smiles.extend(self.hyer_sim['smiles'].tolist())

        runs = []
        for learner in learners:
            for feature in features:
                model = MlModel(algorithm=learner, dataset=dataset, target=target, feat_meth=feature,
                                tune=tune, cv=cv, opt_iter=opt_iter)
                model.featurize()
                model.data_split(val=0.1, add_molecule_to_testset=test_smiles)
                model.reg()
                model.run()
                runs.append({'model_name': model.run_name, 'pred': model.predictions})

        for run in runs:
            run_name = run['model_name']
            pred_results = run['pred']

            for smiles in self.rdkit_sim['smiles'].tolist():
                pred_error = \
                    pred_results.loc[pred_results['smiles'] == smiles].to_dict('records')[0]['pred_average_error']
                self.rdkit_results.append({'smiles': smiles, 'run_name': run_name, 'pred_error': pred_error})

            for smiles in self.jaccard_sim['smiles'].tolist():
                pred_error = \
                    pred_results.loc[pred_results['smiles'] == smiles].to_dict('records')[0]['pred_average_error']
                self.jaccard_results.append({'smiles': smiles, 'run_name': run_name, 'pred_error': pred_error})

            for smiles in self.hyer_sim['smiles'].tolist():
                pred_error = \
                    pred_results.loc[pred_results['smiles'] == smiles].to_dict('records')[0]['pred_average_error']
                self.hyer_results.append({'smiles': smiles, 'run_name': run_name, 'pred_error': pred_error})

            pred_error = pred_results.loc[pred_results['smiles'] ==
                                          self.control_smiles].to_dict('records')[0]['pred_average_error']
            self.control_results.append({'smiles': self.control_smiles, 'run_name': run_name, 'pred_error': pred_error})

    def export_results(self, results_directory):
        if len(os.listdir(results_directory)) != 0:
            raise FileExistsError("\nTarget directory to output results is not empty,"
                                  " trying running self.delete_current_results()")
        pd.DataFrame(self.control_results).to_csv(f'{results_directory}/control_smiles.csv')
        pd.DataFrame(self.rdkit_results).to_csv(f'{results_directory}/rdkit_smiles.csv')
        pd.DataFrame(self.jaccard_results).to_csv(f'{results_directory}/jaccard_smiles.csv')
        pd.DataFrame(self.hyer_results).to_csv(f'{results_directory}/hyer_smiles.csv')
        self.compare_sim_results.to_csv(f'{results_directory}/compare.csv')


if __name__ == "__main__":

    input("Press anything to run")
    results_folders = "results"

    if os.path.exists(results_folders):
        delete_current_results(results_folders)
        os.rmdir(results_folders)
    os.mkdir(results_folders)

    check_for_results_folder(results_directory=results_folders)
    file = "recommender_test_files/lipo_raw.csv"
    raw_data = pd.read_csv(file)
    for i in range(1):
        control_smiles = random.choice(raw_data['smiles'].tolist())
        rec = Recommender(smiles=control_smiles)
        rec.connect_to_neo4j()
        rec.insert_molecules_into_neo4j(dataset=file)
        rec.gather_similar_molecules()
        rec.run_models(dataset=file, target='exp', learners=['rf'], features=[[0]])
        rec.export_results(results_directory=f"{results_folders}/run_{str(i)}")

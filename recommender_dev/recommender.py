import os
import random
import shutil
import time


from difflib import SequenceMatcher
from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from py2neo import Graph, ClientError

from core import MlModel
from core.storage import cd, pickle_model, unpickle_model
from core.features import canonical_smiles

params = {'port': "bolt://localhost:7687", 'username': "neo4j", 'password': "password"}


def cleanup_smiles(file):
    df = pd.read_csv(file)
    df = canonical_smiles(df)
    df.to_csv(file, index=False)


def delete_current_neo4j_data():
    graph = Graph(params['port'], username=params['username'], password=params['password'])

    nodes_to_removes = ['MLModel', 'TestSet', 'TrainSet', 'ValSet']

    for node_type in nodes_to_removes:
        graph.evaluate(f"""
        
            MATCH (n:{node_type})
            OPTIONAL MATCH (n)-[r]-()
            DELETE n, r
        
        """)


def delete_results():
    for directory in os.listdir('results'):
        shutil.rmtree(f'recommender_dev/recommender_test_files/results/{directory}')


def generate_models(dataset, sim_smiles, pulled_smiles):

    print('\nRemoving previous models...')
    for file in os.listdir('recommender_test_files/models'):
        os.remove(f'recommender_dev/recommender_test_files/models/{file}')

    print('\nGenerating models...')
    target = 'exp'

    tune = False
    cv = 5
    opt_iter = 100

    learners = ['svm', 'rf', 'ada', 'gdb']
    features = [[0], [0, 1], [0, 2]]
    # learners = ['rf']
    # features = [[0]]

    print(f'List of learners: {str(learners)}')
    print(f'List of features: {str(features)}')
    print(f'Number of models to run: {len(learners) * len(features)}')

    test_smiles = sim_smiles
    test_smiles.extend([pulled_smiles])

    with cd('recommender_test_files'):
        for learner in learners:
            for feature in features:
                print(f'\nRunning model with parameters: algorithm={learner}, feat={feature}, dataset={dataset}, '
                      f'tune={tune}, cv={cv}, opt_iter={opt_iter}')
                model = MlModel(algorithm=learner, dataset=dataset, target=target, feat_meth=feature,
                                tune=tune, cv=cv, opt_iter=opt_iter)
                model.featurize()
                model.data_split(givenSmiles=test_smiles)
                model.reg()
                model.run()
                with cd('models'):
                    pickle_model(model)


def models_to_neo4j():
    print('\nInserting models into Neo4j...')

    with cd('recommender_test_files/models'):
        for file in os.listdir():
            model = unpickle_model(file)
            model.to_neo4j(**params)


def fetch_actual_value(smiles, target, dataset):
    df = pd.read_csv(dataset)
    match = df.loc[df['smiles'] == smiles]
    value = match.to_dict('records')[0][target]
    return value


def return_sorted_models_for_mol(smiles):
    all_results = []
    with cd('recommender_test_files'):
        for file in os.listdir('models'):
            model = unpickle_model(f'models/{file}')
            results = {'smiles': smiles}
            mp = model.predictions
            matches = mp.loc[mp['smiles'] == smiles]
            if len(matches) == 1:
                row = matches.to_dict('records')[0]
                results['model'] = model.run_name
                results['pred_average_error'] = row['pred_average_error']
                results['in_test_set'] = True
            else:
                predicted_value = predict_molecule(model, smiles)
                actual_value = fetch_actual_value(smiles, target=model.target_name, dataset=model.dataset)
                pred_average_error = abs(actual_value - predicted_value)

                results['model'] = model.run_name
                results['pred_average_error'] = pred_average_error
                results['in_test_set'] = False
            all_results.append(results)
        all_results = pd.DataFrame(all_results)
        return all_results


def sort_dfs():
    with cd('recommender_test_files/results'):
        for directory in os.listdir():
            with cd(directory):
                for file in os.listdir():
                    df = pd.read_csv(file)
                    df = df.sort_values(by=['pred_average_error'])
                    df.to_csv(file, index=False)


def loop_sim_smiles(dataset, run, similar_smiles, similar_smiles_dict, pulled_smiles):
    generate_models(dataset=dataset, sim_smiles=similar_smiles, pulled_smiles=pulled_smiles)
    delete_current_neo4j_data()
    # models_to_neo4j()

    for j, similar_smile in enumerate(similar_smiles_dict):
        smiles = similar_smile['smiles']
        results = return_sorted_models_for_mol(smiles)
        results['weight'] = similar_smile['sim_score']
        file_name = f'molecule_{str(j)}.csv'
        results.to_csv(f'recommender_test_files/results/run_{str(run)}/neo4j_{file_name}')

    results = return_sorted_models_for_mol(pulled_smiles)
    results.to_csv(f'recommender_test_files/results/run_{str(run)}/neo4j_pulled_smiles.csv')
    sort_dfs()


def main():
    dataset = 'lipo_raw.csv'
    data = pd.read_csv(f'recommender_dev/recommender_test_files/{dataset}')

    # cleanup_smiles(dataset)
    # insert_dataset_molecules(data)
    delete_current_neo4j_data()
    delete_results()

    random.seed(5)
    starting_smiles_index = random.randint(0, len(data))

    for run in range(10):

        print(f"\nWorking on run {run}...")
        time.sleep(3)
        os.mkdir(f'recommender_dev/recommender_test_files/results/run_{str(run)}')

        pulled_smiles = data.to_dict('records')[starting_smiles_index + run]['smiles']
        similar_smiles_dict, similar_smiles = find_similar_molecules(pulled_smiles)

        test = similar_smiles
        test.extend([pulled_smiles])

        loop_sim_smiles(dataset=dataset, run=run, similar_smiles=similar_smiles, similar_smiles_dict=similar_smiles_dict,
                        pulled_smiles=pulled_smiles)


if __name__ == '__main__':

    sm = SequenceMatcher(None, '1234', '1324').ratio()
    # main()

import os
import random
import shutil
import time

from tqdm import tqdm
import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import FragmentCatalog, MolFromSmiles
from py2neo import Graph, ClientError
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

from core import MlModel
from core.storage import cd, pickle_model, unpickle_model
from core.features import canonical_smiles

params = {'port': "bolt://localhost:7687", 'username': "neo4j", 'password': "password"}


def cleanup_smiles(dataset):
    file = f'recommender_test_files/{dataset}'
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
    for directory in os.listdir('recommender_test_files/results'):
        shutil.rmtree(f'recommender_test_files/results/{directory}')


def generate_models(dataset, sim_smiles, pulled_smiles):

    print('\nRemoving previous models...')
    for file in os.listdir('recommender_test_files/models'):
        os.remove(f'recommender_test_files/models/{file}')

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


def calculate_fragments(smiles):
    fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
    fparams = FragmentCatalog.FragCatParams(0, 4, fName)  # I need more research and tuning on this one
    fcat = FragmentCatalog.FragCatalog(fparams)  # The fragments are stored as entries
    fcgen = FragmentCatalog.FragCatGenerator()
    mol = MolFromSmiles(smiles)
    fcount = fcgen.AddFragsFromMol(mol, fcat)
    frag_list = []
    for frag in range(fcount):
        frag_list.append(fcat.GetEntryDescription(frag))  # List of molecular fragments
    return frag_list


def insert_dataset_molecules(data):
    print('\ninsert_dataset_molecules...')
    print('This will take some time and only needs to be run once after creating a new DB')
    print('Highly recommended to delete Neo4j data after running this function')

    graph = Graph(params['port'], username=params['username'], password=params['password'])

    try:
        graph.evaluate("CREATE CONSTRAINT ON (n:Molecule) ASSERT n.smiles IS UNIQUE")
        graph.evaluate("CREATE CONSTRAINT ON (n:Fragment) ASSERT n.name IS UNIQUE")
    except ClientError:
        pass

    print('Calculating Fragments...')
    rows = []
    for row in tqdm(data.to_dict('records')):
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


def find_similar_molecules(smiles):
    graph = Graph(params['port'], username=params['username'], password=params['password'])

    print('\nFinding top 5 similar molecules...')

    results = graph.run("""
    
        MATCH (n:Molecule {smiles: $smiles})

        MATCH (m:Molecule)
        WHERE m.smiles <> n.smiles
        
        MATCH (n)-[:HAS_FRAGMENT]->(frag:Fragment)
        WITH collect(frag.name) as fragments_1, n, m
        
        MATCH (m)-[:HAS_FRAGMENT]->(frag:Fragment)
        WITH collect(frag.name) as fragments_2, fragments_1, n, m
        
        MATCH (n)-[:HAS_FRAGMENT]->(frag:Fragment)<-[:HAS_FRAGMENT]-(m)
        WITH collect(frag.name) as fragments_both, fragments_1, fragments_2, n.smiles as smiles_1, m.smiles as smiles_2
        
        RETURN smiles_1, smiles_2, fragments_1, fragments_2, fragments_both
    
    """, parameters={'smiles': smiles}).data()

    sim_results = []
    sim_smiles = []
    for row in results:
        sim_smiles.append(row['smiles_2'])
        both = len(row['fragments_both'])
        total = len(row['fragments_1']) + len(row['fragments_2'])
        row['sim_score'] = 2 * both / total
        sim_results.append(row)
    sim_results = pd.DataFrame(sim_results)
    sim_results = sim_results.sort_values(by=['sim_score'], ascending=False)
    sim_results = sim_results.head(5)
    sim_results = sim_results[['smiles_2', 'sim_score']]
    sim_results = sim_results.rename(columns={'smiles_2': 'smiles'})

    sim_smiles = sim_results['smiles'].to_list()
    sim_results = sim_results.to_dict('records')
    return sim_results, sim_smiles


def predict_molecule(model, smiles):
    feature = model.feat_meth

    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    df = pd.DataFrame([{'smiles': smiles}])

    feat_sets = ['rdkit2d', 'rdkitfpbits', 'morgan3counts', 'morganfeature3counts', 'morganchiral3counts',
                 'atompaircounts']
    selected_feat = [feat_sets[i] for i in feature]
    generator = MakeGenerator(selected_feat)

    columns = []
    for name, numpy_type in generator.GetColumns():
        columns.append(name)

    feature_data = list(map(generator.process, df['smiles']))
    feature_data = pd.DataFrame(feature_data, columns=columns)
    feature_data = feature_data.dropna()
    feature_data = feature_data.drop(list(feature_data.filter(regex='_calculated')), axis=1)
    feature_data = feature_data.drop(list(feature_data.filter(regex='[lL]og[pP]')), axis=1)

    predicted_value = model.estimator.predict(feature_data)[0]
    return predicted_value


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
    data = pd.read_csv(f'recommender_test_files/{dataset}')

    # cleanup_smiles(dataset)
    # insert_dataset_molecules(data)
    delete_current_neo4j_data()
    delete_results()

    random.seed(5)
    starting_smiles_index = random.randint(0, len(data))

    for run in range(10):

        print(f"\nWorking on run {run}...")
        time.sleep(3)
        os.mkdir(f'recommender_test_files/results/run_{str(run)}')

        pulled_smiles = data.to_dict('records')[starting_smiles_index + run]['smiles']
        similar_smiles_dict, similar_smiles = find_similar_molecules(pulled_smiles)

        test = similar_smiles
        test.extend([pulled_smiles])

        loop_sim_smiles(dataset=dataset, run=run, similar_smiles=similar_smiles, similar_smiles_dict=similar_smiles_dict,
                        pulled_smiles=pulled_smiles)


if __name__ == '__main__':
    main()

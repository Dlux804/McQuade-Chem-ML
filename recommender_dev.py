import os

import pandas as pd
from rdkit import Chem, RDConfig
from rdkit.Chem import FragmentCatalog, MolFromSmiles
from py2neo import Graph
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator

from core import MlModel
from core.storage import cd, pickle_model, unpickle_model
from core.features import canonical_smiles

params = {'port': "bolt://localhost:7687", 'username': "neo4j", 'password': "password"}


def test_data():
    print('Verifying starting data...')
    raw_df = pd.read_csv('recommender_test_files/lipo_raw.csv')
    sub_df = pd.read_csv('recommender_test_files/lipo_subset.csv')
    pull_mol = pd.read_csv('recommender_test_files/pulled_molecule.csv')
    smiles = pull_mol.to_dict('records')[0]['smiles']

    matches_in_raw = len(raw_df.loc[raw_df['smiles'] == smiles])
    matches_in_subset = len(sub_df.loc[sub_df['smiles'] == smiles])

    if matches_in_raw == 1 and matches_in_subset == 0:
        print('Valid starting data')
    else:
        raise TypeError('Invalid starting data')


def cleanup_smiles():
    datasets = ['lipo_subset.csv', 'pulled_molecule.csv']

    for dataset in datasets:
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


def generate_models(sim_smiles):
    print('\nRemoving previous models...')
    with cd('recommender_test_files/models'):
        for file in os.listdir():
            os.remove(file)

    print('\nGenerating models...')
    dataset = 'lipo_subset.csv'
    target = 'exp'

    tune = False
    cv = 5
    opt_iter = 100

    # learners = ['svm', 'rf', 'ada', 'gdb']
    # features = [[0], [0, 1], [0, 2]]
    learners = ['rf']
    features = [[0]]

    print(f'List of learners: {str(learners)}')
    print(f'List of features: {str(features)}')
    print(f'Number of models to run: {len(learners) * len(features)}')

    with cd('recommender_test_files'):
        for learner in learners:
            for feature in features:
                print(f'\nRunning model with parameters: algorithm={learner}, feat={feature}, dataset={dataset}, '
                      f'tune={tune}, cv={cv}, opt_iter={opt_iter}')
                model = MlModel(algorithm=learner, dataset=dataset, target=target, feat_meth=feature,
                                tune=tune, cv=cv, opt_iter=opt_iter)
                model.featurize()
                model.data_split(givenSmiles=sim_smiles)
                model.reg()
                model.run()
                with cd('models'):
                    pickle_model(model)


def models_to_neo4j():
    print('\nInserting models into Neo4j...')
    print('If this is the first model being inserted, it may take some time')

    with cd('recommender_test_files/models'):
        for file in os.listdir():
            model = unpickle_model(file)
            model.to_neo4j(**params)


def insert_single_molecule_with_frags(smiles):
    graph = Graph(params['port'], username=params['username'], password=params['password'])

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

    fragments = calculate_fragments(smiles)

    graph.evaluate("""
    
        MERGE (mol:Molecule {smiles: $smiles})
        WITH mol
        UNWIND $fragments as fragment
            MERGE (frag:Fragment {name: fragment})
            MERGE (mol)-[:HAS_FRAGMENT]->(frag)
    
    """, parameters={'smiles': smiles, 'fragments': fragments})


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
    matches = df.loc[df['smiles'] == smiles]
    if len(matches) == 1:
        value = matches.to_dict('records')[0][target]
    else:
        df = pd.read_csv('pulled_molecule.csv')
        matches = df.loc[df['smiles'] == smiles]
        value = matches.to_dict('records')[0][target]
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


def insert_dataset_molecules():
    with cd('recommender_test_files'):

        print('\ninsert_dataset_molecules: Bootleg strats to get molecules into Neo4j...')
        print('Highly recommended to delete Neo4j data after running this function')
        print('This will take some time. This function only needs to be run once after booting a new DB.')

        dataset = 'lipo_subset.csv'
        target = 'exp'

        model = MlModel(algorithm='rf', dataset=dataset, target=target, feat_meth=[0],
                        tune=False, cv=0, opt_iter=0)
        model.featurize()
        if model.algorithm == 'nn':
            model.data_split(val=0.1)
        else:
            model.data_split()
        model.reg()
        model.run()
        model.to_neo4j(**params)


def sort_dfs():
    with cd('recommender_test_files/results'):
        for file in os.listdir():
            df = pd.read_csv(file)
            df = df.sort_values(by=['pred_average_error'])
            df.to_csv(file, index=False)


# test_data()
# cleanup_smiles()
# insert_dataset_molecules()
# delete_current_neo4j_data()

pulled_smiles = 'COc1ccc(CC(=O)Nc2nc3ccccc3[nH]2)cc1'
insert_single_molecule_with_frags(pulled_smiles)
similar_smiles_dict, similar_smiles = find_similar_molecules(pulled_smiles)

results = return_sorted_models_for_mol(pulled_smiles)
results.to_csv(f'recommender_test_files/results/pulled_smiles.csv', index=False)

generate_models(sim_smiles=similar_smiles)
models_to_neo4j()

for i, similar_smile in enumerate(similar_smiles_dict):
    smiles = similar_smile['smiles']
    results = return_sorted_models_for_mol(smiles)
    results['weight'] = similar_smile['sim_score']
    file_name = f'molecule_{str(i)}.csv'
    results.to_csv(f'recommender_test_files/results/{file_name}', index=False)

sort_dfs()

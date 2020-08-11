import json
import os
import re
import shutil
import zipfile
import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from py2neo import Graph, ClientError

from core.storage.misc import __clean_up_param_grid_item__, NumpyEncoder
from core.storage.dictionary import target_name_grid
from core.neo4j.fragments import calculate_fragments

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class ModelOrOutputToNeo4j:

    def __init__(self, model=None, zipped_out_dir=None, molecules_per_batch=5000, port="bolt://localhost:7687",
                 username="neo4j", password="password"):
        if model is None and zipped_out_dir is None:
            raise Exception("Must specify weather model or zipped_out_dir")
        if model is not None and zipped_out_dir is not None:
            raise Exception("Cannot be both a model and zipped output dir")

        self.batch = molecules_per_batch
        self.json_data = None
        self.model_data = None
        self.graph = Graph(port, username=username, password=password)

        if model is not None:

            # Use slightly modified script of store() used for model outputs

            all_raw_dfs = {}

            d = dict(vars(model))  # create dict of model attributes
            objs = []  # empty lists to capture excepted attributes
            for k, v in d.items():

                # grab pandas related objects for export to csv
                if isinstance(v, pd.core.frame.DataFrame) or isinstance(v, pd.core.series.Series):
                    if k != "smiles_series":
                        all_raw_dfs[k] = v

                if k == 'param_grid' or k == 'params':  # Param grid does not behave properly,
                    new_param_grid_dict = {}
                    for label, item in d[k].items():
                        # Cleanup each item in param_gird dict
                        new_param_grid_dict[label] = __clean_up_param_grid_item__(item)
                    d[k] = new_param_grid_dict

                if k == 'fit_params' and not isinstance(v, type(None)):
                    epochs = v['epochs']
                    d.update({'max_epochs': epochs})
                    del d[k]

                # grab non-Java compatible attributes
                if not isinstance(v, (int, float, dict, tuple, list, bool, str, type(None))):
                    objs.append(k)

            # reduce list of exceptions to unique entries
            for k in objs:
                d.pop(k)

            self.json_data = json.loads(json.dumps(d, cls=NumpyEncoder))
            self.model_data = all_raw_dfs['data']

        else:  # zipped_out_dir=True

            safety_string = '$$temp$$output$$dont$$copy$$this$$name$$:)$$'

            with zipfile.ZipFile(str(zipped_out_dir), 'r') as zip_ref:
                zip_ref.extractall(safety_string)

            for file in os.listdir(safety_string):
                split_file = file.split('.')
                file_name = split_file[0]

                file_type = file_name.split('_')
                file_type = file_type[len(file_type) - 1]

                if file_type == 'data':
                    self.model_data = pd.read_csv(f'{safety_string}/{file}')
                if file_type == 'attributes':
                    with open(f'{safety_string}/{file}', 'r') as f:
                        json_data = f.read()
                    self.json_data = json.loads(json_data)

            shutil.rmtree(safety_string, ignore_errors=True)

        if self.model_data is None:
            raise Exception("Could not parse model data")
        if self.json_data is None:
            raise Exception("Could not parse json data")

        # Here for reference if you want to view attributes stored in json file

        # for label, value in self.json_data.items():
        #     print(label, value)

        if not self.json_data['tuned']:
            self.tune_algorithm_name = str(None)
        else:
            self.tune_algorithm_name = self.json_data['tune_algorithm_name']

        test_data = self.model_data.loc[self.model_data['in_set'] == 'test']
        train_data = self.model_data.loc[self.model_data['in_set'] == 'train']
        val_data = self.model_data.loc[self.model_data['in_set'] == 'val']
        self.spilt_data = {'TestSet': test_data, 'TrainSet': train_data, 'ValSet': val_data}

        self.check_for_constraints()
        self.create_main_nodes()
        self.merge_molecules_with_sets()
        self.merge_molecules_with_dataset()
        self.merge_molecules_with_frags()
        self.merge_molecules_with_feats()
        self.merge_feats_with_rdkit2d()

    def check_for_constraints(self):

        constraint_check_strings = [
            """
        CREATE CONSTRAINT ON (n:Model)
        ASSERT n.name IS UNIQUE
        """,

            """
        CREATE CONSTRAINT ON (n:FeatureList)
        ASSERT n.feat_IDs IS UNIQUE
        """,

            """
        CREATE CONSTRAINT ON (n:Dataset)
        ASSERT n.data IS UNIQUE
        """,

            """
        CREATE CONSTRAINT ON (n:Fragment) 
        ASSERT n.name IS UNIQUE        
        """,

            """
        CREATE CONSTRAINT ON (n:Molecule) 
        ASSERT n.smiles IS UNIQUE    
        """,

            """
        CREATE CONSTRAINT ON (n:Feature) 
        ASSERT n.name IS UNIQUE      
        """,

            """
        CREATE CONSTRAINT ON (n:TuningAlg) 
        ASSERT n.algorithm IS UNIQUE      
        """,

            """
        CREATE CONSTRAINT ON (n:Algorithm) 
        ASSERT n.name IS UNIQUE      
        """,

            """
        CREATE CONSTRAINT ON (n:Set) 
        ASSERT n.name IS UNIQUE      
        """,

            """
        CREATE CONSTRAINT ON (n:FeatureMethod) 
        ASSERT n.name IS UNIQUE      
        """
        ]

        for constraint_check_string in constraint_check_strings:
            try:
                self.graph.evaluate(constraint_check_string)
            except ClientError:
                pass

    def create_main_nodes(self):

        js = self.json_data

        self.graph.evaluate(
            """
        
            MERGE (model:Model {name: $model_name})
            ON CREATE SET model.data = $date, model.feat_time = $feat_time, model.test_time = $test_time,
                model.train_time = $train_time, model.seed = $seed, model.test_percent = $test_percent,
                model.train_percent = $train_percent, model.val_percent = $val_percent
        
                MERGE (dataset:Dataset {data: $data})
                    ON CREATE SET dataset.size = $dataset_size, dataset.target = $target, dataset.source = $source,
                        dataset.task_type = $task_type
                        
                    MERGE (model)-[:uses_dataset]->(dataset)
                    
                MERGE (tuning:TuningAlg {algorithm: $tune_algorithm_name})
                    MERGE (model)-[:tuned_with]->(tuning)
                    
                MERGE (algo:Algorithm {name: $algo_name, source: 'sklearn'})
                    MERGE (model)-[:uses_algorithm]->(algo)
            
                MERGE (featlist:FeatureList {feat_IDs: $feat_IDs})
                    MERGE (model)-[:featurized_by]->(featlist)
                    WITH featlist
                    UNWIND $feature_methods as feature_method
                        MERGE (featmeth:FeatureMethod {name: feature_method})
                        MERGE (featlist)-[:contains_feature_method]->(featmeth)
        
            """,
            parameters={'date': js['date'], 'feat_time': js['feat_time'], 'model_name': js['run_name'],
                        'test_time': float(js['predictions_stats']['time_avg']),
                        'train_time': js['tune_time'], 'seed': js['random_seed'],
                        'test_percent': js['test_percent'], 'train_percent': js['train_percent'],
                        'val_percent': js['val_percent'],

                        'data': js['dataset'], 'dataset_size': js['n_tot'], 'target': js['target_name'],
                        'source': 'MolecularNetAI', 'task_type': js['task_type'],

                        'tune_algorithm_name': self.tune_algorithm_name,

                        'algo_name': js['algorithm'],

                        'feat_IDs': js['feat_meth'],
                        'feature_methods': js['feat_method_name'],
                        }
        )

    def molecule_query_loop(self, molecules, query, **params):
        range_molecules = []
        for index, molecule in enumerate(molecules):
            range_molecules.append(molecule)
            if index % self.batch == 0 and index != 0:
                self.graph.evaluate(query, parameters={'molecules': range_molecules, **params})
                range_molecules = []
        if range_molecules:
            self.graph.evaluate(query, parameters={'molecules': range_molecules, **params})

    def merge_molecules_with_sets(self):

        for datatype, df in self.spilt_data.items():
            if len(df) > 0:
                df = df[['smiles', self.json_data['target_name']]]
                df = df.rename(columns={self.json_data['target_name']: 'target'})
                molecules = df.to_dict('records')

                target_name_for_neo4j = target_name_grid(self.json_data['dataset'])
                size = len(molecules)
                rel_dict = {'TrainSet': 'trained_with', 'TestSet': 'predicts', 'Valset': 'validates'}

                query = """
                        MATCH (model:Model {name: $run_name})
                        MERGE (set:Set {run_name: $run_name, name: $set_type})
                            ON CREATE SET set.size = $size
                        MERGE (model)-[:%s]->(set)

                        WITH set
                        UNWIND $molecules as molecule
                            MERGE (mol:Molecule {smiles: molecule.smiles})
                                SET mol.%s = molecule.target
                            MERGE (set)-[:contains_molecule]->(mol)

                        """ % (rel_dict[datatype], target_name_for_neo4j)

                self.molecule_query_loop(molecules, query, target=self.json_data['target_name'],
                                         run_name=self.json_data['run_name'], set_type=datatype, size=size)

    def merge_molecules_with_dataset(self):

        df = self.model_data[['smiles', self.json_data['target_name']]]
        df = df.rename(columns={self.json_data['target_name']: 'target'})
        molecules = df.to_dict('records')

        target_name_for_neo4j = target_name_grid(self.json_data['dataset'])

        query = """
                                    
                MATCH (dataset:Dataset {data: $dataset})
                UNWIND $molecules as molecule
                    MERGE (mol:Molecule {smiles: molecule.smiles})
                        SET mol.%s = molecule.target
                    MERGE (dataset)-[:contains_molecule]->(mol)
                    
                """ % target_name_for_neo4j

        self.molecule_query_loop(molecules, query, dataset=self.json_data['dataset'])

    def merge_molecules_with_frags(self):
        df = self.model_data[['smiles']]
        df['fragments'] = df['smiles'].apply(calculate_fragments)
        molecules = df.to_dict('records')

        query = """
                
                UNWIND $molecules as molecule
                    MERGE (mol:Molecule {smiles: molecule.smiles})
                        FOREACH (fragment in molecule.fragments |
                            MERGE (frag:Fragment {name: fragment})
                            MERGE (mol)-[:has_fragment]->(frag)
                            )

                """

        self.molecule_query_loop(molecules, query)

    def merge_molecules_with_feats(self):

        if 'rdkit2d' not in self.json_data['feat_method_name']:
            return

        feats = [val for val in self.json_data['feature_list'] if re.search(r'fr_|Count|Num|Charge|TPSA|qed|%s', val)]
        feats = self.model_data[feats]

        feat_dicts = []
        for index, row in feats.iterrows():
            row_feat_dicts = []
            row = dict(row)
            for label, value in row.items():
                row_feat_dicts.append({'name': label, 'value': value})
            feat_dicts.append(row_feat_dicts)

        df = self.model_data[['smiles']]
        df['feats'] = feat_dicts
        molecules = df.to_dict('records')

        query = """
                
                UNWIND $molecules as molecule
                    MERGE (mol:Molecule {smiles: molecule.smiles})
                    FOREACH (feature in molecule.feats |
                        MERGE (feat:Feature {name: feature.name})
                        MERGE (mol)-[:has_descriptor {value: feature.value}]->(feat)
                        )

                """

        self.molecule_query_loop(molecules, query)

    def merge_feats_with_rdkit2d(self):

        if 'rdkit2d' not in self.json_data['feat_method_name']:
            return

        feats = [feat for feat in self.json_data['feature_list']
                 if re.search(r'fr_|Count|Num|Charge|TPSA|qed|%s', feat)]

        query = """
        
                MERGE (feature_meth:FeatureMethod {name: 'rdkit2d'})
                WITH feature_meth
                UNWIND $feats as feat
                    MERGE (feature:Feature {name: feat}) 
                    MERGE (feature_meth)-[:uses_feature_method]->(feature)

                """

        self.graph.evaluate(query, parameters={'feats': feats})

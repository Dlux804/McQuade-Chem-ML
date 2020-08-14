import json
import os
import re
import shutil
import zipfile
import warnings

import pandas as pd
from numpy import nan
from pandas.core.common import SettingWithCopyWarning
from py2neo import Graph, ClientError
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog, MolFromSmiles

from core.storage.misc import __clean_up_param_grid_item__, NumpyEncoder
from core.storage.dictionary import target_name_grid

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class ModelOrOutputToNeo4j:

    def __init__(self, model=None, zipped_out_dir=None, qsar_obj=None, molecules_per_batch=5000,
                 port="bolt://localhost:7687", username="neo4j", password="password"):

        self.batch = molecules_per_batch
        self.model = model
        self.zipped_out_dir = zipped_out_dir
        self.qsar_obj = qsar_obj
        self.graph = Graph(port, username=username, password=password)

        self.verify_input_variables()

        self.parsed_models = None

        if model is not None:
            self.parsed_models = self.initialize_model_object()

        elif zipped_out_dir is not None:
            self.parsed_models = self.initialize_output_dir()

        elif qsar_obj is not None:
            self.parsed_models = self.initialize_qsar_obj()

        for model in self.parsed_models:
            self.model_data = model['model_data']
            self.json_data = model['json_data']

            if self.model_data is None:
                raise ValueError("Could not parse model data")
            if self.json_data is None:
                raise ValueError("Could not parse json data")

            # Here for reference if you want to view attributes stored in json file

            # for label, value in self.json_data.items():
            #     print(label, value)

            if not self.json_data['tuned']:
                self.tune_algorithm_name = None
            else:
                self.tune_algorithm_name = self.json_data['tune_algorithm_name']

            test_data = self.model_data.loc[self.model_data['in_set'] == 'test']
            train_data = self.model_data.loc[self.model_data['in_set'] == 'train']
            val_data = self.model_data.loc[self.model_data['in_set'] == 'val']
            self.spilt_data = {'TestSet': test_data, 'TrainSet': train_data, 'ValSet': val_data}

            self.check_for_constraints()
            self.create_main_nodes()
            self.merge_model_with_tuning()
            self.merge_featlist_with_model()
            self.merge_feats_with_rdkit2d()

            # If dataset does not exist in neo4j graph
            if not self.check_for_dataset():
                self.merge_molecules_with_sets()
                self.merge_molecules_with_dataset()
                self.merge_molecules_with_frags()
                self.merge_molecules_with_feats()

    def verify_input_variables(self):
        conflicting_variables = [self.model, self.zipped_out_dir, self.qsar_obj]
        found_non_none_type_var = False
        for conflicting_variable in conflicting_variables:
            if conflicting_variable is not None and found_non_none_type_var is True:
                raise ValueError("Multiple objects (model_object, output_dir, or qsar_dir) "
                                 "to initialize inputted, please only specify one")
            elif conflicting_variable is not None:
                found_non_none_type_var = True
        if not found_non_none_type_var:
            raise ValueError("Object to input into Neo4j not found, please specify model_object,"
                             "output_dir, or qsar_dir")

    def initialize_model_object(self):
        # Use slightly modified script of store() used for model outputs

        all_raw_dfs = {}

        d = dict(vars(self.model))  # create dict of model attributes
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

        json_data = json.loads(json.dumps(d, cls=NumpyEncoder))
        model_data = all_raw_dfs['data']
        json_data['is_qsarDB'] = False
        json_data['source'] = 'MolecularNetAI'
        model = [{'model_data': model_data, 'json_data': json_data}]
        return model

    def initialize_output_dir(self):
        json_data = None
        model_data = None
        safety_string = '$$temp$$output$$dont$$copy$$this$$name$$:)$$'

        with zipfile.ZipFile(str(self.zipped_out_dir), 'r') as zip_ref:
            zip_ref.extractall(safety_string)

        for file in os.listdir(safety_string):
            split_file = file.split('.')
            file_name = split_file[0]

            file_type = file_name.split('_')
            file_type = file_type[len(file_type) - 1]

            if file_type == 'data':
                model_data = pd.read_csv(f'{safety_string}/{file}')
            if file_type == 'attributes':
                with open(f'{safety_string}/{file}', 'r') as f:
                    json_data = f.read()
                json_data = json.loads(json_data)

        shutil.rmtree(safety_string, ignore_errors=True)
        json_data['is_qsarDB'] = False
        json_data['source'] = 'MolecularNetAI'
        model = [{'model_data': model_data, 'json_data': json_data}]
        return model

    def initialize_qsar_obj(self):
        models = []
        for model in self.qsar_obj.models:
            model_data = model.raw_data
            json_data = {'test_percent': round(model.n['testing'] / model.n_total, 2),
                         'train_percent': round(model.n['training'] / model.n_total, 2),
                         'val_percent': round(model.n['validation'] / model.n_total, 2),
                         'predictions_stats': {'r2_avg': model.r2, 'mse_avg': model.mse,
                                               'rmse_avg': model.rmse,
                                               'time_avg': None},
                         'run_name': model.name,
                         'feature_list': self.qsar_obj.feats,
                         'algorithm': model.algorithm,
                         'task_type': model.task_type,
                         'target_name': model.target_name,
                         'n_tot': model.n_total,
                         'dataset': self.qsar_obj.dataset,
                         'date': None,
                         'feat_time': None,
                         'tune_time': None,
                         'random_seed': None,
                         'tuned': False,
                         'feat_meth': None,
                         'feat_method_name': [],
                         'is_qsarDB': True,
                         'source': 'QsarDB'}
            models.append({'model_data': model_data, 'json_data': json_data})
        return models

    def check_for_constraints(self):

        node_unique_prop_dict = {'MLModel': 'name',
                                 'Algorithm': 'name',
                                 'RandomSpilt': 'run_name',
                                 'DataSet': 'data',
                                 'TestSet': 'run_name',
                                 'TrainSet': 'run_name',
                                 'ValSet': 'run_name',
                                 'Tuning': 'name',
                                 'FeatureList': 'feat_IDs',
                                 'FeatureMethod': 'name',
                                 'Feature': 'name',
                                 'Fragment': 'name',
                                 'Molecule': 'smiles'
                                 }

        for node, prop in node_unique_prop_dict.items():

            constraint_string = """
                        CREATE CONSTRAINT ON (n:%s)
                        ASSERT n.%s IS UNIQUE
                        """ % (node, prop)
            try:
                self.graph.evaluate(constraint_string)
            except ClientError:
                pass

    def create_main_nodes(self):

        js = self.json_data

        self.graph.evaluate(
            """
        
            MERGE (model:MLModel {name: $model_name})
            ON CREATE SET model.date = $date, model.feat_time = $feat_time, model.test_time = $test_time,
                model.train_time = $train_time, model.seed = $seed

                MERGE (spilt:RandomSpilt {run_name: $model_name})
                    ON CREATE SET spilt.train_percent = $train_percent, spilt.test_percent = $test_percent, 
                        spilt.val_percent = $val_percent 
                    MERGE (model)-[:USES_SPLIT]->(spilt)
        
                MERGE (dataset:DataSet {data: $data})
                    ON CREATE SET dataset.size = $dataset_size, dataset.target = $target, dataset.source = $source,
                        dataset.task_type = $task_type
                    MERGE (model)-[:USES_DATASET]->(dataset)
                    MERGE (spilt)-[:SPLITS_DATASET]->(dataset)
                    
                MERGE (testset:TestSet {run_name: $model_name, name: 'TestSet'})
                MERGE (dataset)-[:SPILTS_INTO_TEST]->(testset)
                MERGE (spilt)-[:MAKES_SPLIT]->(testset)
                
                MERGE (trainset:TrainSet {run_name: $model_name, name: 'TrainSet'})
                MERGE (dataset)-[:SPILTS_INTO_TRAIN]->(trainset)
                MERGE (spilt)-[:MAKES_SPLIT]->(trainset)
                
                MERGE (valset:ValSet {run_name: $model_name, name: 'ValSet'})
                MERGE (dataset)-[:SPILTS_INTO_VAL]->(valset)
                MERGE (spilt)-[:MAKES_SPLIT]->(valset)
                    
                MERGE (algo:Algorithm {name: $algo_name, source: 'sklearn'})
                    MERGE (model)-[:USES_ALGORITHM]->(algo)
        
            """,
            parameters={'date': js['date'], 'feat_time': js['feat_time'], 'model_name': js['run_name'],
                        'test_time': js['predictions_stats']['time_avg'],
                        'train_time': js['tune_time'], 'seed': js['random_seed'],
                        'test_percent': js['test_percent'], 'train_percent': js['train_percent'],
                        'val_percent': js['val_percent'],
                        'tune_algorithm_name': self.tune_algorithm_name,

                        'data': js['dataset'], 'dataset_size': js['n_tot'], 'target': js['target_name'],
                        'source': js['source'], 'task_type': js['task_type'],

                        'algo_name': js['algorithm'],

                        'feat_IDs': js['feat_meth'],
                        'feature_methods': js['feat_method_name'],
                        }
        )

    def merge_model_with_tuning(self):

        if self.tune_algorithm_name is not None:
            self.graph.evaluate(
                """
        
                MATCH (model:MLModel {name: $model_name}) 
                MERGE (tuning:Tuning {name: $tune_algorithm_name})
                MERGE (model)-[:USES_TUNING]->(tuning)
                
                """,
                parameters={'model_name': self.json_data['run_name'],
                            'tune_algorithm_name': self.tune_algorithm_name}
            )

    def merge_featlist_with_model(self):

        js = self.json_data

        if not self.json_data['is_qsarDB']:
            self.graph.evaluate(
                """
            
                MATCH (model:MLModel {name: $model_name})
                MERGE (featlist:FeatureList {feat_IDs: $feat_IDs})
                    MERGE (model)-[:USES_FACTORIZATION]->(featlist)
                    WITH featlist, model
                    UNWIND $feature_methods as feature_method
                        MERGE (featmeth:FeatureMethod {name: feature_method})
                        MERGE (featmeth)-[:CONTRIBUTES_TO]->(featlist)
                        MERGE (model)-[:USES_FACTORIZATION]->(featmeth)
            
                """,
                parameters={'model_name': js['run_name'],
                            'feat_IDs': js['feat_meth'],
                            'feature_methods': js['feat_method_name']
                            }
            )

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
                    MERGE (feature_meth)-[:CALCULATES]->(feature)

                """

        self.graph.evaluate(query, parameters={'feats': feats})

    def check_for_dataset(self):

        check = self.graph.evaluate(
            """
            MATCH (dataset:DataSet {data: $data})-[:CONTAINS_MOLECULE]->(:Molecule)
            RETURN dataset LIMIT 1
            """,
            parameters={'data': self.json_data['dataset']}
        )
        if check is not None:
            return True
        return False

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

            # Gather data
            if datatype == 'TestSet' and not self.json_data['is_qsarDB']:
                pred_stats = self.json_data['predictions_stats']
                r2_avg = pred_stats['r2_avg']
                r2_std = pred_stats['r2_std']
                mse_avg = pred_stats['mse_avg']
                mse_std = pred_stats['mse_std']
                rmse_avg = pred_stats['rmse_avg']
                rmse_std = pred_stats['rmse_std']

                scaled_pred_stats = self.json_data['scaled_predictions_stats']
                scaled_r2_avg = scaled_pred_stats['r2_avg_scaled']
                scaled_r2_std = scaled_pred_stats['r2_std_scaled']
                scaled_mse_avg = scaled_pred_stats['mse_avg_scaled']
                scaled_mse_std = scaled_pred_stats['mse_std_scaled']
                scaled_rmse_avg = scaled_pred_stats['rmse_avg_scaled']
                scaled_rmse_std = scaled_pred_stats['rmse_std_scaled']
            elif self.json_data['is_qsarDB']:
                rd = {'TrainSet': 'training', 'TestSet': 'testing', 'ValSet': 'validation'}
                r2_avg = self.json_data['predictions_stats']['r2_avg'][rd[datatype]]
                mse_avg = self.json_data['predictions_stats']['mse_avg'][rd[datatype]]
                rmse_avg = self.json_data['predictions_stats']['rmse_avg'][rd[datatype]]

                rmse_std = mse_std = r2_std = None
                scaled_r2_avg = scaled_mse_avg = scaled_rmse_avg = None
                scaled_r2_std = scaled_mse_std = scaled_rmse_std = None
            else:
                r2_std = mse_std = rmse_std = None
                r2_avg = mse_avg = rmse_avg = None
                scaled_r2_avg = scaled_r2_std = scaled_mse_avg = scaled_mse_std = None
                scaled_rmse_avg = scaled_rmse_std = None

            # If dataset exists
            if len(df) > 0:
                df = df[['smiles', self.json_data['target_name']]]
                df = df.rename(columns={self.json_data['target_name']: 'target'})
                molecules = df.to_dict('records')

                target_name_for_neo4j = target_name_grid(self.json_data['dataset'])
                if target_name_for_neo4j is None:
                    target_name_for_neo4j = self.json_data['target_name']
                size = len(molecules)
                rel_dict = {'TrainSet': 'TRAINS', 'TestSet': 'PREDICTS', 'ValSet': 'VALIDATES'}
                mol_dataset_dict = {'TrainSet': 'CONTAINS_TRAINED_MOLECULE',
                                    'TestSet': 'CONTAINS_PREDICTED_MOLECULE',
                                    'ValSet': 'CONTAINS_VALIDATED_MOLECULE'}

                query = """
                        MATCH (model:MLModel {name: $run_name})
                        MATCH (set:%s {run_name: $run_name, name: $set_type})
                        
                        MERGE (model)-[set_rel:`%s`]->(set)
                            ON CREATE SET set_rel.size = $size, set_rel.r2_avg = $r2_avg, set_rel.r2_std = $r2_std, 
                                set_rel.mse_avg = $mse_avg, set_rel.mse_std = $mse_std, set_rel.rmse_avg = $rmse_avg, 
                                set_rel.rmse_std = $rmse_std,
                                
                                set_rel.scaled_r2_avg = $scaled_r2_avg, set_rel.scaled_mse_avg = $scaled_mse_avg,
                                set_rel.scaled_rmse_avg = $scaled_rmse_avg, set_rel.scaled_r2_std = $scaled_r2_std,
                                set_rel.scaled_mse_std = $scaled_mse_std, set_rel.scaled_rmse_std = $scaled_rmse_std

                        WITH set
                        UNWIND $molecules as molecule
                            MERGE (mol:Molecule {smiles: molecule.smiles})
                                SET mol.`%s` = molecule.target
                            MERGE (set)-[:%s]->(mol)

                        """ % (datatype, rel_dict[datatype], target_name_for_neo4j, mol_dataset_dict[datatype])
                self.molecule_query_loop(molecules, query, target=self.json_data['target_name'],
                                         run_name=self.json_data['run_name'], set_type=datatype, size=size,

                                         r2_avg=r2_avg, r2_std=r2_std, mse_avg=mse_avg, mse_std=mse_std,
                                         rmse_avg=rmse_avg, rmse_std=rmse_std,

                                         scaled_r2_avg=scaled_r2_avg, scaled_mse_avg=scaled_mse_avg,
                                         scaled_rmse_avg=scaled_rmse_avg, scaled_r2_std=scaled_r2_std,
                                         scaled_mse_std=scaled_mse_std, scaled_rmse_std=scaled_rmse_std
                                         )

    def merge_molecules_with_dataset(self):

        df = self.model_data[['smiles', self.json_data['target_name']]]
        df = df.rename(columns={self.json_data['target_name']: 'target'})
        molecules = df.to_dict('records')

        target_name_for_neo4j = target_name_grid(self.json_data['dataset'])
        if target_name_for_neo4j is None:
            target_name_for_neo4j = self.json_data['target_name']

        query = """
                                    
                MATCH (dataset:DataSet {data: $dataset})
                UNWIND $molecules as molecule
                    MERGE (mol:Molecule {smiles: molecule.smiles})
                        SET mol.`%s` = molecule.target
                    MERGE (dataset)-[:CONTAINS_MOLECULE]->(mol)
                    
                """ % target_name_for_neo4j
        self.molecule_query_loop(molecules, query, dataset=self.json_data['dataset'])

    def merge_molecules_with_frags(self):

        def calculate_fragments(smiles):
            """
            Objective: Create fragments and import them into Neo4j based on our ontology
            Intent: This script is based on Adam's "mol_frag.ipynb" file in his deepml branch, which is based on rdkit's
                    https://www.rdkit.org/docs/GettingStartedInPython.html. I still need some council on this one since we can
                    tune how much fragment this script can generate for one SMILES. Also, everything (line 69 to 77)
                    needs to be under a for loop or else it will break (as in not generating the correct amount of fragments,
                    usually much less than the actual amount). I'm not sure why
            :param smiles:
            :return:
            """
            fName = os.path.join(RDConfig.RDDataDir, 'FunctionalGroups.txt')
            fparams = FragmentCatalog.FragCatParams(0, 4, fName)  # I need more research and tuning on this one
            fcat = FragmentCatalog.FragCatalog(fparams)  # The fragments are stored as entries
            fcgen = FragmentCatalog.FragCatGenerator()
            mol = MolFromSmiles(smiles)
            fcount = fcgen.AddFragsFromMol(mol, fcat)
            # print("This SMILES, %s, has %d fragments" % (smiles, fcount))
            frag_list = []
            for frag in range(fcount):
                frag_list.append(fcat.GetEntryDescription(frag))  # List of molecular fragments
            return frag_list

        df = self.model_data[['smiles']]
        df['fragments'] = df['smiles'].apply(calculate_fragments)
        molecules = df.to_dict('records')

        query = """
                
                UNWIND $molecules as molecule
                    MERGE (mol:Molecule {smiles: molecule.smiles})
                        FOREACH (fragment in molecule.fragments |
                            MERGE (frag:Fragment {name: fragment})
                            MERGE (mol)-[:HAS_FRAGMENT]->(frag)
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
                        MERGE (mol)-[:HAS_DESCRIPTOR {value: feature.value}]->(feat)
                        )

                """
        self.molecule_query_loop(molecules, query)

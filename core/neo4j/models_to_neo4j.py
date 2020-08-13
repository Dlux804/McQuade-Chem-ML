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
            raise ValueError("Object to input into Neo4j not found, please speciiy model_object,"
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
        CREATE CONSTRAINT ON (n:Algorithm) 
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
            ON CREATE SET model.date = $date, model.feat_time = $feat_time, model.test_time = $test_time,
                model.train_time = $train_time, model.seed = $seed
        
                // Split node is a hidden node, no property to merge on
                MERGE (model)-[:uses_split]->(:Split {train_percent: $train_percent, test_percent: $test_percent,
                    val_percent: $val_percent})
                    
                MERGE (tuning:Tuning {name: $tune_algorithm_name})
                    MERGE (model)-[:tuned_with]->(tuning)
        
                MERGE (dataset:Dataset {data: $data})
                    ON CREATE SET dataset.size = $dataset_size, dataset.target = $target, dataset.source = $source,
                        dataset.task_type = $task_type
                    MERGE (model)-[:uses_dataset]->(dataset)
                    
                MERGE (algo:Algorithm {name: $algo_name, source: 'sklearn'})
                    MERGE (model)-[:uses_algorithm]->(algo)
        
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

    def merge_featlist_with_model(self):

        js = self.json_data

        if not self.json_data['is_qsarDB']:

            self.graph.evaluate(
                """
            
                MATCH (model:Model {name: $model_name})
                MERGE (featlist:FeatureList {feat_IDs: $feat_IDs})
                    MERGE (model)-[:featurized_by]->(featlist)
                    WITH featlist
                    UNWIND $feature_methods as feature_method
                        MERGE (featmeth:FeatureMethod {name: feature_method})
                        MERGE (featlist)-[:contains_feature_method]->(featmeth)
            
                """,
                parameters={'model_name': js['run_name'],
                            'feat_IDs': js['feat_meth'],
                            'feature_methods': js['feat_method_name']
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

            if datatype == 'TestSet' and not self.json_data['is_qsarDB']:
                r2_avg = self.json_data['predictions_stats']['r2_avg']
                r2_std = self.json_data['predictions_stats']['r2_std']
                mse_avg = self.json_data['predictions_stats']['mse_avg']
                mse_std = self.json_data['predictions_stats']['mse_std']
                rmse_avg = self.json_data['predictions_stats']['rmse_avg']
                rmse_std = self.json_data['predictions_stats']['rmse_std']
            elif self.json_data['is_qsarDB']:
                rd = {'TrainSet': 'training', 'TestSet': 'testing', 'ValSet': 'validation'}
                r2_avg = self.json_data['predictions_stats']['r2_avg'][rd[datatype]]
                mse_avg = self.json_data['predictions_stats']['mse_avg'][rd[datatype]]
                rmse_avg = self.json_data['predictions_stats']['rmse_avg'][rd[datatype]]
                rmse_std = None
                mse_std = None
                r2_std = None
            else:
                r2_avg = r2_std = mse_avg = mse_std = rmse_avg = rmse_std = None

            if len(df) > 0:
                df = df[['smiles', self.json_data['target_name']]]
                df = df.rename(columns={self.json_data['target_name']: 'target'})
                molecules = df.to_dict('records')

                target_name_for_neo4j = target_name_grid(self.json_data['dataset'])
                if target_name_for_neo4j is None:
                    target_name_for_neo4j = self.json_data['target_name']
                size = len(molecules)
                rel_dict = {'TrainSet': 'trained_with', 'TestSet': 'predicts', 'ValSet': 'validated_by'}

                query = """
                        MATCH (model:Model {name: $run_name})
                        MERGE (set:Set {run_name: $run_name, name: $set_type})
                            ON CREATE SET set.size = $size, set.r2_avg = $r2_avg, set.r2_std = $r2_std,
                                set.mse_avg = $mse_avg, set.mse_std = $mse_std, set.rmse_avg = $rmse_avg,
                                set.rmse_std = $rmse_std
                        MERGE (model)-[:`%s`]->(set)

                        WITH set
                        UNWIND $molecules as molecule
                            MERGE (mol:Molecule {smiles: molecule.smiles})
                                SET mol.`%s` = molecule.target
                            MERGE (set)-[:contains_molecule]->(mol)

                        """ % (rel_dict[datatype], target_name_for_neo4j)

                self.molecule_query_loop(molecules, query, target=self.json_data['target_name'],
                                         run_name=self.json_data['run_name'], set_type=datatype, size=size,
                                         r2_avg=r2_avg, r2_std=r2_std, mse_avg=mse_avg, mse_std=mse_std,
                                         rmse_avg=rmse_avg, rmse_std=rmse_std)

    def merge_molecules_with_dataset(self):

        df = self.model_data[['smiles', self.json_data['target_name']]]
        df = df.rename(columns={self.json_data['target_name']: 'target'})
        molecules = df.to_dict('records')

        target_name_for_neo4j = target_name_grid(self.json_data['dataset'])
        if target_name_for_neo4j is None:
            target_name_for_neo4j = self.json_data['target_name']

        query = """
                                    
                MATCH (dataset:Dataset {data: $dataset})
                UNWIND $molecules as molecule
                    MERGE (mol:Molecule {smiles: molecule.smiles})
                        SET mol.`%s` = molecule.target
                    MERGE (dataset)-[:contains_molecule]->(mol)
                    
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

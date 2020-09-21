import json
import os
import re
import zipfile
import warnings
import ast

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from py2neo import Graph, ClientError
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog, MolFromSmiles

from core.storage.misc import __clean_up_param_grid_item__, NumpyEncoder
from core.storage.dictionary import target_name_grid
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


class ModelToNeo4j:

    def __init__(self, model=None, zipped_out_dir=None, qsar_obj=None, molecules_per_batch=1000,
                 port="bolt://localhost:7687", username="neo4j", password="password"):

        """

        The point of this file is to simply our Neo4j scripts. The logic for how data is inserted into Neo4j has
        not changed from our previous versions. Rather this is meant to act as a more easily to follow version
        of what we had before. This is a completely inclusive file that will take a model object, zipped output
        directory, or qsar_obj (from QsarDB zipped directory) and directly pipe it into Neo4j.

        The goal of this file was to simply data import, and add code readability

        :param model: A model object, mostly called from withen model object, model.to_neo4j(**params)
        :param zipped_out_dir:  A created output directory from pipeline.
        :param qsar_obj: A QsarDB qdb database object generated from qsar_to_neo4j.py
        :param molecules_per_batch: How many molecules to insert in batches (recommended not to exceed 1,000, highly
                                    recommended not to exceed 4,000, 5,000 and over will break code)
        :param port: Port to connect to for neo4j
        :param username: Username for Neo4j
        :param password: Password
        """

        """
            actual = pva['actual']
            pva_predictions = pva.drop(['pred_avg', 'pred_std', 'smiles', 'actual'], axis=1)
            average_error = list(pva_predictions.sub(actual, axis=0).mean(axis=1))  # Calculate avg prediction error
        """

        self.batch = molecules_per_batch
        self.model = model
        self.zipped_out_dir = zipped_out_dir
        self.qsar_obj = qsar_obj
        self.graph = Graph(port, username=username, password=password)
        self.parsed_models = None
        self.molecule_stats = ['pred_average_error', 'pred_std', 'pred_avg']  # Define stats to pull out of predictions

        # Make sure variable do not clash
        self.verify_input_variables()

        # Parse data depending on input
        if model is not None:
            self.parsed_models = self.initialize_model_object()
        elif zipped_out_dir is not None:
            self.parsed_models = self.initialize_output_dir()
        elif qsar_obj is not None:
            self.parsed_models = self.initialize_qsar_obj()

        # Insert each model (QsarDBs can have multiple models in them)
        for model in self.parsed_models:
            self.model_data = model['model_data']
            self.json_data = model['json_data']

            if self.model_data is None:
                raise ValueError("Could not parse model data")
            if self.json_data is None:
                raise ValueError("Could not parse json data")

            # Here for reference if you want to view attributes stored in json file

            for label, value in self.json_data.items():
                print(label, value)

            if not self.json_data['tuned']:
                self.tune_algorithm_name = None
            else:
                self.tune_algorithm_name = self.json_data['tune_algorithm_name']

            if 'params' in self.json_data.keys():
                self.params = self.json_data['params']
            else:
                self.params = {}

            if isinstance(self.json_data['target_name'], list):
                if 'smiles' in self.json_data['target_name']:
                    self.json_data['target_name'].remove('smiles')

            # Gather test/train/val data into organized bins
            test_data = self.model_data.loc[self.model_data['in_set'] == 'test']
            train_data = self.model_data.loc[self.model_data['in_set'] == 'train']
            val_data = self.model_data.loc[self.model_data['in_set'] == 'val']
            self.split_data = {'TestSet': test_data, 'TrainSet': train_data, 'ValSet': val_data}

            # Generate and merge core nodes (Very fast)
            self.check_for_constraints()
            self.create_main_nodes()
            self.merge_model_with_algorithm()
            self.merge_model_with_tuning()
            self.merge_featlist_and_featmeths_with_model()
            self.merge_feats_with_rdkit2d()
            self.merge_molecules_with_sets()

            # If dataset does not exist in neo4j graph, merge molecules and fragments (sorta slow, 5 secs to 7 mins)
            # if not self.check_for_dataset():
            self.merge_molecules_with_dataset()
            self.merge_molecules_with_frags()
            self.merge_molecules_with_feats()

    def verify_input_variables(self):
        """

        A model object, zipped_output_dir, and/or QsarDB can not be passed at the same time. Make sure that only
        one is being passed. And make sure that a object of some sort is being passed

        :return:
        """

        conflicting_variables = [self.model, self.zipped_out_dir, self.qsar_obj]  # Variables to check
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
        """

        The goal is to have all the data being piped to Neo4j be the exact same. It is a lot easier to format
        the model object data to the output data rather than the other way around (how we were doing it before).
        Our current store() is basically what is being done here, but is slightly modified to not actually create
        output files

        :return:
        """
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
        model_data = self.__combine_model_data_with_predictions__(all_raw_dfs['data'], all_raw_dfs['predictions'],
                                                                  all_raw_dfs['scaled_predictions'])

        json_data['is_qsarDB'] = False
        json_data['source'] = 'MolecularNetAI'
        model = [{'model_data': model_data,
                  'json_data': json_data}]

        return model

    def initialize_output_dir(self):
        """

        The data above was formatted to this data. All that is being done here is pulling out the attibutes.json
        and predictions.csv files and saved.

        This function is not optimal, as the entire directory is extracted before pulling out the json and csv file.
        We should find a way to pull out specific files out of a zipped directory

        :return:
        """

        def __pull_data__(file_tp):  # Get csv data from zipped_dir, return pandas df
            zipped_dir.extract(file_tp)  # Only file that gets extracted
            file_tp = current_dir + '/' + file_tp
            data = pd.read_csv(file)
            os.remove(file_tp)  # Remove extracted file
            return data

        # Place holders
        json_data = None
        model_data = None

        current_dir = os.getcwd()
        with zipfile.ZipFile(str(self.zipped_out_dir), 'r') as zipped_dir:

            # Read each file name in zipped_dir
            for file in zipped_dir.namelist():

                # Get file name, drop extension
                split_file = file.split('.')
                file_name = split_file[0]

                # Get the name of the end part of file in the directory
                file_type = file_name.split('_')
                if file_type[len(file_type) - 2] == 'scaled':
                    file_type = 'scaled_' + file_type[len(file_type) - 1]
                else:
                    file_type = file_type[len(file_type) - 1]

                if file_type == 'data':
                    model_data = __pull_data__(file)

                if file_type == 'attributes':
                    json_data = zipped_dir.read(file)
                    json_data = json.loads(json_data)

                if file_type == 'predictions':
                    predictions_csv_data = __pull_data__(file)

                if file_type == 'scaled_predictions':
                    scaled_predictions_csv_data = __pull_data__(file)

        # model_data = self.__combine_model_data_with_predictions__(model_data, predictions_csv_data,
        #                                                           scaled_predictions_csv_data)

        json_data['is_qsarDB'] = False
        json_data['source'] = 'MolecularNetAI'
        model = [{'model_data': model_data,
                  'json_data': json_data}]

        return model

    def __combine_model_data_with_predictions__(self, model_data, predictions, scaled_predictions):

        def __merge_pred_x_with_data__(data, pred_data, pred_column):
            # Merge data with pred_column, replace numpy NaNs with None, for Neo4j
            data = data.merge(pred_data, on='smiles', how='outer')
            data[pred_column] = data[pred_column].where(pd.notnull(data[pred_column]), None)
            return data

        for molecule_stat in self.molecule_stats:
            # Gather stat in predictions, merge with model data
            stat_column = predictions[['smiles', molecule_stat]]
            model_data = __merge_pred_x_with_data__(model_data, stat_column, molecule_stat)

            # Gather stat in scaled predictions, merge with model data
            scaled_stat_column = scaled_predictions[['smiles', molecule_stat]]
            scaled_stat_column = scaled_stat_column.rename(columns={molecule_stat: f'scaled_{molecule_stat}'})
            model_data = __merge_pred_x_with_data__(model_data, scaled_stat_column, f'scaled_{molecule_stat}')

        return model_data

    def initialize_qsar_obj(self):
        """

        The messiest way we have to import data. QsarDB data is very different from our current data. A lot of love
        was put in to format to our current output files. This function is just pulling out the information that is
        generated in qsar_to_neo4j.py

        :return:
        """

        models = []
        for model in self.qsar_obj.models:

            # Get raw model_data
            model_data = model.raw_data

            # Create dumpy columns to cast None for each molecule in model_data
            model_data['pred_MSE'] = None
            model_data['scaled_pred_MSE'] = None
            model_data['pred_RMSE'] = None
            model_data['scaled_pred_RMSE'] = None
            model_data['pred_std'] = None
            model_data['scaled_pred_std'] = None

            # Gather pseudo json_data
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
                         'feat_meth': None,
                         'cv': None,
                         'opt_iter': None,
                         'n_best': None,
                         'delta': None,

                         'tuned': False,
                         'feat_method_name': [],
                         'is_qsarDB': True,
                         'source': 'QsarDB'
                         }

            models.append({'model_data': model_data,
                           'json_data': json_data})
        return models

    def check_for_constraints(self):

        """

        This function makes sure that all the nodes have their proper constraints

        :return:
        """

        node_unique_prop_dict = {'MLModel': 'name',
                                 'Algorithm': 'name',
                                 'RandomSplit': 'run_name',
                                 'DataSet': 'data',
                                 'TestSet': 'run_name',
                                 'TrainSet': 'run_name',
                                 'ValSet': 'run_name',
                                 'Tuning': 'name',
                                 'FeatureList': 'features',
                                 'FeatureMethod': 'name',
                                 'Feature': 'name',
                                 'Fragment': 'name',
                                 'Molecule': 'smiles'
                                 }

        for node, prop in node_unique_prop_dict.items():

            constraint_string = f"""
                                CREATE CONSTRAINT ON (n:{node})
                                ASSERT n.{prop} IS UNIQUE
                                """
            try:
                self.graph.evaluate(constraint_string)
            except ClientError:
                pass

    def create_main_nodes(self):

        """

        This is probably the most important function, and the one we will likely modify with time. This function
        creates all the main nodes and the relationships to and from them. Here we can easily add and drop relationships
        to these core nodes.

        Please do not merge molecules or features in this functions :)

        :return:
        """

        js = self.json_data

        self.graph.evaluate(
            """
        
            MERGE (model:MLModel {name: $model_name})
            ON CREATE SET model.date = $date, model.feat_time = $feat_time, model.test_time = $test_time,
                model.train_time = $train_time, model.seed = $seed

                MERGE (split:RandomSplit {run_name: $model_name})
                    ON CREATE SET split.train_percent = $train_percent, split.test_percent = $test_percent, 
                        split.val_percent = $val_percent, split.seed = $seed, split.name = "RandomSplit"
                    MERGE (model)-[:USES_SPLIT]->(split)
        
                MERGE (dataset:DataSet {data: $data})
                    ON CREATE SET dataset.size = $dataset_size, dataset.target = $target, dataset.source = $source,
                        dataset.task_type = $task_type, dataset.name = "DataSet"
                    MERGE (model)-[:USES_DATASET]->(dataset)
                    MERGE (split)-[:SPLITS_DATASET]->(dataset)
                    
                MERGE (testset:TestSet {run_name: $model_name, name: 'TestSet'})
                MERGE (dataset)-[:SPLITS_INTO_TEST]->(testset)
                MERGE (split)-[:MAKES_SPLIT]->(testset)
                
                MERGE (trainset:TrainSet {run_name: $model_name, name: 'TrainSet'})
                MERGE (dataset)-[:SPLITS_INTO_TRAIN]->(trainset)
                MERGE (split)-[:MAKES_SPLIT]->(trainset)
                
                MERGE (valset:ValSet {run_name: $model_name, name: 'ValSet'})
                MERGE (dataset)-[:SPLITS_INTO_VAL]->(valset)
                MERGE (split)-[:MAKES_SPLIT]->(valset)
        
            """,
            parameters={'date': js['date'], 'feat_time': js['feat_time'], 'model_name': js['run_name'],
                        'test_time': js['predictions_stats']['time_avg'],
                        'train_time': js['tune_time'], 'seed': js['random_seed'],
                        'test_percent': js['test_percent'], 'train_percent': js['train_percent'],
                        'val_percent': js['val_percent'],
                        'tune_algorithm_name': self.tune_algorithm_name,

                        'data': js['dataset'], 'dataset_size': js['n_tot'], 'target': js['target_name'],
                        'source': js['source'], 'task_type': js['task_type'],

                        'features': js['feat_meth'],
                        'feature_methods': js['feat_method_name'],
                        }
        )

    def merge_model_with_algorithm(self):

        for label, value in self.params.items():
            if isinstance(value, str):
                value = f"'{value}'"
            self.graph.evaluate(f"""

                                MATCH (model:MLModel {'{name: $model_name}'})

                                MERGE (algo:Algorithm {'{name: $algo_name, source: "sklearn"}'})
                                MERGE (model)-[algo_rel:USES_ALGORITHM]->(algo)
                                    SET algo_rel.{label} = "{value}"

                                """,
                                parameters={'model_name': self.json_data['run_name'],
                                            'algo_name': self.json_data['algorithm'],
                                            }
                                )
        if len(self.params) == 0:
            self.graph.evaluate("""
                                MATCH (model:MLModel {name: $model_name})

                                MERGE (algo:Algorithm {name: $algo_name, source: "sklearn"})
                                MERGE (model)-[algo_rel:USES_ALGORITHM]->(algo)
                                """,
                                parameters={'model_name': self.json_data['run_name'],
                                            'algo_name': self.json_data['algorithm'],
                                            }
                                )

    def merge_model_with_tuning(self):

        """

        Separate function to merge the model to the tuning nodes. This was separate because it is possible that
        models are not tuned, and we want to makes sure a node isn't created if the model is not merged.

        :return:
        """

        if self.tune_algorithm_name is not None:
            self.graph.evaluate(
                """
        
                MATCH (model:MLModel {name: $model_name}) 
                MERGE (tuning:Tuning {name: $tune_algorithm_name})
                MERGE (model)-[tuning_rel:USES_TUNING]->(tuning)
                    ON CREATE SET tuning_rel.cv = $cv, tuning_rel.opt_iter = $opt_iter, 
                                  tuning_rel.tune_time = $tune_time, tuning_rel.delta = $delta,
                                  tuning_rel.n_best = $n_best
                    ON CREATE SET tuning_rel += $cv_results
                
                """,
                parameters={'model_name': self.json_data['run_name'],
                            'tune_algorithm_name': self.tune_algorithm_name,

                            'cv': self.json_data['cv_folds'],
                            'opt_iter': self.json_data['opt_iter'],
                            'tune_time': self.json_data['tune_time'],
                            'n_best': self.json_data['cp_n_best'],
                            'delta': self.json_data['cp_delta'],
                            'cv_results': (self.json_data['cv_results'])

                            }
            )

    def merge_featlist_and_featmeths_with_model(self):

        js = self.json_data

        if not self.json_data['is_qsarDB']:
            self.graph.evaluate(
                """
            
                MATCH (model:MLModel {name: $model_name})
                MERGE (featlist:FeatureList {features: $features})
                    ON CREATE SET featlist.name = "FeatureList"
                    MERGE (model)-[:USES_FEATURE_LIST]->(featlist)
                    WITH featlist, model
                    UNWIND $feature_methods as feature_method
                        MERGE (featmeth:FeatureMethod {name: feature_method})
                        MERGE (featmeth)-[:USED_BY_FEATURE_LIST]->(featlist)
                        MERGE (model)-[:USES_FEATURE_METHOD]->(featmeth)
            
                """,
                parameters={'model_name': js['run_name'],
                            'features': js['feat_meth'],
                            'feature_methods': js['feat_method_name']
                            }
            )

    def merge_feats_with_rdkit2d(self):

        if 'rdkit2d' not in self.json_data['feat_method_name']:  # Note how this is only done for rdkit2d (Sorry Qsar)
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

    def molecule_query_loop(self, molecules, query, **params):

        """
        Loop for molecule queries. We do not want to merge 14k molecules at once, that is sure way to kill a computer.
        So a set batch number is set, and we do not insert more molecules at once than the batch number allows for.

        :param molecules:
        :param query:
        :param params:
        :return:
        """

        range_molecules = []
        for index, molecule in enumerate(molecules):
            range_molecules.append(molecule)
            if index % self.batch == 0 and index != 0:
                self.graph.evaluate(query, parameters={'molecules': range_molecules, **params})
                range_molecules = []
        if range_molecules:
            self.graph.evaluate(query, parameters={'molecules': range_molecules, **params})

    def merge_molecules_with_sets(self):

        # TODO spilt this into three different functions, one for TestSet, TrainSet, ValSet. This is hard to follow

        """

        This will generate the data in the relationships relating models to Train/Test/Val sets. As well as the
        molecules inside of those sets. How the rmse, mse, r2 is calculate for each dataset differs slighty, so
        that has to be accounted for. Plus many models may not have a test/val set, so that has to be considered
        as well.

        :return:
        """

        for datatype, df in self.split_data.items():

            # Gather data
            # if datatype == 'TestSet' and not self.json_data['is_qsarDB']:
            if None is not None:
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
                molecule_stats = []
                # molecule_stats = ['scaled_' + x for x in self.molecule_stats]
                # molecule_stats.extend(self.molecule_stats)

                target_names = self.json_data['target_name']

                if not isinstance(target_names, list):
                    target_names = [target_names]

                for target_name in target_names:
                    sub_df = df[['smiles', target_name, *molecule_stats]]
                    sub_df = sub_df.rename(columns={target_name: 'target'})
                    molecules = sub_df.to_dict('records')

                    target_name_for_neo4j = target_name_grid(self.json_data['dataset'])
                    if target_name_for_neo4j is None:
                        target_name_for_neo4j = self.json_data['target_name']
                    size = len(molecules)

                    # Dict relating Set to the relationship name needed for the relationship (Model)->(Set)
                    rel_dict = {'TrainSet': 'TRAINS',
                                'TestSet': 'PREDICTS',
                                'ValSet': 'VALIDATES'}

                    # Dict relating molecules to the relationship name needed for relationship (Set)->(Molecule)
                    mol_dataset_dict = {'TrainSet': 'CONTAINS_TRAINED_MOLECULE',
                                        'TestSet': 'CONTAINS_PREDICTED_MOLECULE',
                                        'ValSet': 'CONTAINS_VALIDATED_MOLECULE'}

                    query = f"""
                            MATCH (model:MLModel {'{name: $run_name}'})
                            MATCH (set:{datatype} {'{run_name: $run_name, name: $set_type}'})
                            
                            MERGE (model)-[set_rel:`{rel_dict[datatype]}`]->(set)
                                ON CREATE SET set_rel.size = $size, set_rel.r2_avg = $r2_avg, set_rel.r2_std = $r2_std, 
                                    set_rel.mse_avg = $mse_avg, set_rel.mse_std = $mse_std, 
                                    set_rel.rmse_avg = $rmse_avg, set_rel.rmse_std = $rmse_std,
                                    
                                    set_rel.scaled_r2_avg = $scaled_r2_avg, set_rel.scaled_mse_avg = $scaled_mse_avg,
                                    set_rel.scaled_rmse_avg = $scaled_rmse_avg, set_rel.scaled_r2_std = $scaled_r2_std,
                                    set_rel.scaled_mse_std = $scaled_mse_std, set_rel.scaled_rmse_std = $scaled_rmse_std
    
                            WITH set
                            UNWIND $molecules as molecule
                                MERGE (mol:Molecule {'{smiles: molecule.smiles}'})
                                    SET mol.`{target_name_for_neo4j}` = molecule.target
                                MERGE (set)-[mol_rel:{mol_dataset_dict[datatype]}]->(mol)
                                    SET mol_rel.average_error = molecule.pred_average_error,
                                        mol_rel.uncertainty = molecule.pred_std,
                                        mol_rel.predicted_average = molecule.pred_avg,
    
                                        mol_rel.scaled_average_error = molecule.scaled_pred_average_error,
                                        mol_rel.scaled_uncertainty = molecule.scaled_pred_std,
                                        mol_rel.scaled_predicted_average = molecule.scaled_pred_avg,
    
                                        mol_rel.actual_value = molecule.target
                                    
                            """
                    self.molecule_query_loop(molecules, query, target=target_name,
                                             run_name=self.json_data['run_name'], set_type=datatype, size=size,

                                             r2_avg=r2_avg, r2_std=r2_std, mse_avg=mse_avg, mse_std=mse_std,
                                             rmse_avg=rmse_avg, rmse_std=rmse_std,

                                             scaled_r2_avg=scaled_r2_avg, scaled_mse_avg=scaled_mse_avg,
                                             scaled_rmse_avg=scaled_rmse_avg, scaled_r2_std=scaled_r2_std,
                                             scaled_mse_std=scaled_mse_std, scaled_rmse_std=scaled_rmse_std,
                                             )

    def check_for_dataset(self):

        """

        This function will check and see if the dataset is already in the database. If it is, then we dont want
        to have to re-calculate the fragments and features. While this does not take too, too long. It is still
        good to save time if we can

        :return:
        """

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

    def merge_molecules_with_dataset(self):

        target_names = self.json_data['target_name']
        if not isinstance(target_names, list):
            target_names = [target_names]

        for target_name in target_names:

            df = self.model_data[['smiles', target_name]]
            df = df.rename(columns={target_name: 'target'})
            molecules = df.to_dict('records')

            df.to_csv('dev.csv')

            target_name_for_neo4j = target_name_grid(self.json_data['dataset'])
            if target_name_for_neo4j is None:
                target_name_for_neo4j = target_name

            query = f"""
                                        
                    MATCH (dataset:DataSet {'{data: $dataset}'})
                    UNWIND $molecules as molecule
                        MERGE (mol:Molecule {'{smiles: molecule.smiles}'})
                            SET mol.`{target_name_for_neo4j}` = molecule.target
                        MERGE (dataset)-[:CONTAINS_MOLECULE]->(mol)
                        
                    """
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

        """

        Frist we make sure that rdkit2d was used, then merge the molecules to the features with the values of the
        descriptor

        :return:
        """

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

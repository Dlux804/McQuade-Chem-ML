"""
Objective: Import data from _data.csv and _attributes.csv files to create TestSet, ValSet
TrainSet, SMILES, RandomSplit and their relationships with each other in Neo4j
"""

from py2neo import Graph, Node

g = Graph("bolt://localhost:7687", user="neo4j", password="1234")


def smiles_to_neo(smiles, target, data_list):
    """

    :param smiles:
    :param target:
    :param data_list:
    :return:
    """
    record = g.run("""MATCH (n:SMILES {SMILES:"%s"}) RETURN n""" % smiles)
    if len(list(record)) > 0:
        print(f"This SMILES, {smiles}, already exists. Updating its properties and relationships")
        g.evaluate("UNWIND $parameters as rows "
                   "match (mol:SMILES {SMILES: $smiles}), "
                   "(dataset:DataSet {data:rows.dataset}) "
                   "set mol.measurement = mol.measurement + $target, mol.dataset = mol.dataset + rows.dataset",
                   parameters={'parameters': data_list, 'smiles': smiles, 'target': target})
    else:
        mol = Node("SMILES", SMILES=smiles)
        g.merge(mol, "SMILES", "SMILES")
        g.evaluate("UNWIND $parameters as rows "
                   "match (mol:SMILES {SMILES: $smiles}) set mol.measurement = [$target],"
                   "mol.dataset = [rows.dataset]",
                   parameters={'parameters': data_list, 'smiles': smiles, 'target': target})


def split_to_neo(split_df, df_from_attributes, data_list, split):
    """

    :param split_df:
    :param df_from_attributes:
    :param data_list:
    :param split:
    :return:
    """
    r2 = float(df_from_attributes["predictions_stats.r2_avg"])
    rmse = float(df_from_attributes["predictions_stats.rmse_avg"])
    mse = float(df_from_attributes["predictions_stats.mse_avg"])
    length = len(split_df.iloc[:, 0])
    smiles_list = list(split_df['smiles'])
    target_list = list(split_df['target'])

    for smiles, target in zip(smiles_list, target_list):
        if split != "Test":  # For ValSet and TrainSet nodes
            # Create SMILES
            smiles_to_neo(smiles, target, data_list)
            # Create the specific split node
            g.evaluate(""" merge (splitset:%sSet {name:"%sSet", %ssize:$size}) """ % (split, split, split),
                       parameters={'smiles': smiles, 'size': length})
            # Create relationships between SMILES and the specific split node
            g.evaluate("""  match (mol:SMILES {SMILES: $smiles}), (splitset:%sSet {name:"%sSet", %ssize:$size})
                               merge (splitset)-[:CONTAINS_MOLECULES]->(mol)""" % (split, split, split),
                       parameters={'smiles': smiles, 'size': length})
            # Create relationships between RandomSplit node and the specific split node
            g.evaluate(""" UNWIND $parameters as rows match (randomsplit:RandomSplit {name:"RandomSplit", 
                                       test_percent:rows.test_percent, train_percent:rows.train_percent, 
                                       random_seed:rows.random_seed}), (splitset:%sSet {%ssize:$size}) 
                                       merge (randomsplit)-[:SPLITS_INTO]->(splitset)
                                       """ % (split, split), parameters={'parameters': data_list, 'size': length})
        else:  # For TestSet node
            smiles_to_neo(smiles, target, data_list)
            g.evaluate(""" merge (splitset:%sSet {name:"%sSet", %ssize:$size, r2: $r2, rmse: $rmse, mse: $mse}) 
                        """ % (split, split, split),
                       parameters={'smiles': smiles, 'size': length, 'r2': r2, 'mse': mse, 'rmse': rmse})
            g.evaluate("""  match (mol:SMILES {SMILES: $smiles}), (splitset:%sSet {r2: $r2, %ssize:$size})
                            merge (splitset)-[:CONTAINS_MOLECULES]->(mol)""" % (split, split),
                       parameters={'smiles': smiles, 'size': length, 'r2': r2})

            g.evaluate(""" UNWIND $parameters as rows match (randomsplit:RandomSplit {name:"RandomSplit", 
                                       test_percent:rows.test_percent, train_percent:rows.train_percent, 
                                       random_seed:rows.random_seed}), (splitset:%sSet {%ssize:$size, r2: $r2}) 
                                       merge (randomsplit)-[:SPLITS_INTO]->(splitset)
                                       """ % (split, split), parameters={'parameters': data_list, 'size': length,
                                                                         'r2': r2})

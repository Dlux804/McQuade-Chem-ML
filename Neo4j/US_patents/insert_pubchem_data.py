import ast
import os
import timeit
import pandas as pd

import py2neo
from py2neo import Graph, NodeMatcher
import concurrent.futures as cf
from rdkit.Chem import MolToSmiles, MolFromSmiles, MolToSmarts
from rdkit.Chem.Descriptors import MolWt
from Neo4j.BulkChem.backends import get_fragments, get_file_location
import sys
from pubchempy import get_sdf


def gather_compounds():

    graph = Graph()
    matcher = NodeMatcher(graph)
    compounds = matcher.match('compound')

    gathered_compounds = []
    counter = 0
    main_counter = 0
    for compound in compounds:
        gathered_compounds.append(compound)
        counter = counter + 1
        main_counter = main_counter + 1
        if counter >= 100000:
            df = pd.DataFrame.from_records(gathered_compounds)
            df.to_csv('compounds/{}_{}.csv'.format(main_counter-counter, main_counter-1))
            gathered_compounds = []
            counter = 0
    df = pd.DataFrame.from_records(gathered_compounds)
    df.to_csv('compounds/{}_{}.csv'.format(main_counter-counter, main_counter-1), index=False)


def test(testing_data):
    testing_data = ast.literal_eval(testing_data)
    compound_name = testing_data[len(testing_data)-1]
    properties = get_properties('IsomericSMILES', compound_name, 'name')
    return properties


def get_pubchem_data():
    for file in os.listdir('compounds'):
        data = pd.read_csv('compounds/' + file)
        print(data.apply(lambda x: test(x['chemical_names']), axis=1))


properties_list = ['MolecularFormula', 'CanonicalSMILES', 'IsomericSMILES', 'InChIKey', 'IUPACName', 'XLogP',
                   'ExactMass', 'MonoisotopicMass', 'TPSA', 'Complexity', 'Charge', 'HBondDonorCount',
                   'HBondAcceptorCount', 'RotatableBondCount', 'HeavyAtomCount', 'IsotopeAtomCount', 'AtomStereoCount',
                   'DefinedAtomStereoCount', 'UndefinedAtomStereoCount', 'BondStereoCount', 'DefinedBondStereoCount',
                   'UndefinedBondStereoCount', 'CovalentUnitCount', 'Volume3D', 'XStericQuadrupole3D',
                   'YStericQuadrupole3D', 'ZStericQuadrupole3D', 'FeatureCount3D', 'FeatureAcceptorCount3D',
                   'FeatureDonorCount3D', 'FeatureAnionCount3D', 'FeatureCationCount3D', 'FeatureRingCount3D',
                   'FeatureHydrophobeCount3D', 'ConformerModelRMSD3D', 'EffectiveRotorCount3D', 'ConformerCount3D']

#gather_compounds()
#get_pubchem_data()
properties = get_sdf('p-tert-butylphenol', 'name')
print(properties)
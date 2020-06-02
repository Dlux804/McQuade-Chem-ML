import os
import hashlib
import pathlib
import pandas as pd
from rdkit.Chem import rdChemReactions, Draw, MolFromSmiles, MolFromSmarts, MolToSmiles

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def clean_up_checker_files(path_to_directory):
    main_directories = os.listdir(path_to_directory)
    for main_directory in main_directories:
        main_directory = path_to_directory + "/" + main_directory
        for directory in os.listdir(main_directory):
            directory = main_directory + '/' + directory
            for file in os.listdir(directory):
                file = directory + '/' + file
                split_file = file.split('.')
                if split_file[len(split_file) - 1] == 'checker':
                    os.remove(file)


def get_file_location():
    return str(pathlib.Path(__file__).parent.absolute())


def __fgm__(smarts, mol):
    sub_mol = MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(sub_mol))


def __cul__(lis):
    temp = {}
    for pair in lis:
        number_of_matches = pair.split('|')[0]
        group = pair.split('|')[1]
        if group not in temp.keys():
            temp[group] = number_of_matches
        else:
            temp[group] = int(temp[group]) + int(number_of_matches)
    lis = []
    for group, number_of_matches in temp.items():
        lis.append(str(number_of_matches) + "|" + group)
    return lis


def get_functional_groups(smiles, fragments_df=None, as_dict=False):
    smiles_mol = MolFromSmiles(smiles)
    if smiles_mol is None:
        return None

    if fragments_df is None:
        path = pathlib.Path(__file__).parent.absolute()
        fragments_df = pd.read_csv(str(path) + '/datafiles/Function-Groups-SMARTS.csv')

    fragments_df['number_of_matches'] = fragments_df['SMARTS'].apply(__fgm__, mol=smiles_mol)
    fragments_df = fragments_df.loc[fragments_df['number_of_matches'] > 0]

    if as_dict:
        return dict(zip(fragments_df.Fragment, fragments_df.number_of_matches))

    groups = (fragments_df['number_of_matches'].astype(str) + '|' + fragments_df['Fragment']).tolist()
    return __cul__(groups)


def __dts__(a, b):
    lis_a = []
    for group, num in a.items():
        lis_a.append(f'{num}|{group}')
    a = ','.join(lis_a)

    lis_b = []
    for group, num in b.items():
        lis_b.append(f'{num}|{group}')
    b = ','.join(lis_b)
    return f'{a}>{b}'


def __dod__(a, b):

    a_dict = {}
    for group, num in a.items():
        if group in b.keys():
            if num > b[group]:
                a_dict[group] = num - b[group]
        else:
            a_dict[group] = num

    b_dict = {}
    for group, num in b.items():
        if group in a.keys():
            if num > a[group]:
                b_dict[group] = num - a[group]
        else:
            b_dict[group] = num

    return a_dict, b_dict


def get_compounds_functional_groups(compounds, fragments_df=None):
    all_functional_groups = {}
    for compound in compounds:
        functional_groups = get_functional_groups(MolToSmiles(compound), fragments_df=fragments_df, as_dict=True)
        if functional_groups:
            for group, num in functional_groups.items():
                if group in all_functional_groups.keys():
                    all_functional_groups[group] = all_functional_groups[group] + num
                else:
                    all_functional_groups[group] = num
    return all_functional_groups


def map_rxn_functional_groups(reaction_smiles, difference_only=True, fragments_df=None):
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles)

    all_reactant_functional_groups = get_compounds_functional_groups(rxn.GetReactants(), fragments_df=fragments_df)
    all_product_functional_groups = get_compounds_functional_groups(rxn.GetProducts(), fragments_df=fragments_df)

    if not difference_only:
        return __dts__(all_reactant_functional_groups, all_product_functional_groups)

    all_reactant_functional_groups, all_product_functional_groups = __dod__(all_reactant_functional_groups,
                                                                            all_product_functional_groups)

    return __dts__(all_reactant_functional_groups, all_product_functional_groups)


def classify_reaction(reaction_smiles, fragments_df=None):
    mapped_frags = map_rxn_functional_groups(reaction_smiles, fragments_df=fragments_df)
    mapping_file = str(pathlib.Path(__file__).parent.absolute()) + '/datafiles/mapped_reactions.csv'
    mapping_df = pd.read_csv(mapping_file)

    classifiation = mapping_df.loc[mapping_df['mapped_groups'] == mapped_frags].to_dict('record')
    if classifiation:
        return classifiation[0]['reaction_class']

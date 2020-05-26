import os
import hashlib
import pathlib
import pandas as pd
from rdkit.Chem import rdChemReactions, Draw, MolFromSmiles, MolFromSmarts, MolToSmiles


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


def compress(uncompressed_smiles):
    try:
        m = hashlib.sha256()
        m.update(uncompressed_smiles)
        compressed_smiles = m.hexdigest()
    except TypeError:
        m = hashlib.sha256()
        m.update(bytes(uncompressed_smiles, 'UTF-8'))
        compressed_smiles = m.hexdigest()
    return compressed_smiles


def save_reaction_image(smiles, directory_location, svg=False):
    if not os.path.exists(directory_location):
        raise Exception('Directory not found')

    compressed_smiles = compress(smiles)
    rxn = rdChemReactions.ReactionFromSmarts(smiles)

    if svg:
        file_location = f'{directory_location}/{compressed_smiles}.svg'
        rimage = Draw.ReactionToImage(rxn, useSVG=True)
        text_file = open(file_location, "w+")
        text_file.write(rimage)
        text_file.close()
        return file_location

    file_location = f'{directory_location}/{compressed_smiles}.png'
    if os.path.exists(file_location):
        return file_location

    rimage = Draw.ReactionToImage(rxn)
    rimage.save(file_location)
    return file_location


def get_file_location():
    return str(pathlib.Path(__file__).parent.absolute())


def __fgm__(smarts, mol):
    sub_mol = MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(sub_mol))


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


def map_rxn_functional_groups(reaction_smiles, difference_only=True):
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles)
    all_reactant_functional_groups = []
    for reactant in rxn.GetReactants():
        reactant_functional_groups = get_functional_groups(MolToSmiles(reactant))
        for group in reactant_functional_groups:
            all_reactant_functional_groups.append(group)
    all_reactant_functional_groups = __cul__(all_reactant_functional_groups)

    all_product_functional_groups = []
    for product in rxn.GetProducts():
        product_functional_groups = get_functional_groups(MolToSmiles(product))
        for group in product_functional_groups:
            all_product_functional_groups.append(group)
    all_product_functional_groups = __cul__(all_product_functional_groups)

    if not difference_only:
        return ','.join(all_reactant_functional_groups) + '>' + ','.join(all_product_functional_groups)

    a = all_reactant_functional_groups
    b = all_product_functional_groups
    all_reactant_functional_groups = [x for x in a if x not in b]
    all_product_functional_groups = [x for x in b if x not in a]
    return ','.join(all_reactant_functional_groups) + '>' + ','.join(all_product_functional_groups)

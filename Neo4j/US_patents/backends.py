import os
import hashlib
import pathlib
import pandas as pd
from rdkit.Chem import rdChemReactions, Draw, MolFromSmiles, MolFromSmarts


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


def save_reaction_image(smiles, directory_location):
    if not os.path.exists(directory_location):
        raise Exception('Directory not found')

    rxn = rdChemReactions.ReactionFromSmarts(smiles)
    rimage = Draw.ReactionToImage(rxn)
    compressed_smiles = compress(smiles)
    file_location = f'{directory_location}/{compressed_smiles}.png'
    rimage.save(file_location)
    return file_location


def get_file_location():
    return str(pathlib.Path(__file__).parent.absolute())


def test_functional_group(smarts, mol):
    sub_mol = MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(sub_mol))


def get_fragments(smiles, fragments_df=None):
    smiles_mol = MolFromSmiles(smiles)
    if smiles_mol is None:
        return None

    if fragments_df is None:
        path = pathlib.Path(__file__).parent.absolute()
        fragments_df = pd.read_csv(str(path) + '/datafiles/Function-Groups-SMARTS.csv')

    fragments_df['number_of_matches'] = fragments_df['SMARTS'].apply(test_functional_group, mol=smiles_mol)
    fragments_df = fragments_df.loc[fragments_df['number_of_matches'] > 0]
    test = fragments_df['number_of_matches'].astype(str) + '|' + fragments_df['Fragment']
    return test.tolist()
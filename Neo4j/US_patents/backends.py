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


def save_reaction_image(smiles, directory_location='C:/xampp/htdocs', xampp_installed=True):
    if not os.path.exists('C:/xampp/htdocs') and xampp_installed:
        raise Exception('XAMPP is not installed or is set to a directory other than default. Please specify where'
                        'htdocs directory is located, or set xampp_installed=False')
    if not os.path.exists('C:/xampp/htdocs/reactions') and xampp_installed:
        os.mkdir('C:/xampp/htdocs/reactions')

    rxn = rdChemReactions.ReactionFromSmarts(smiles)
    rimage = Draw.ReactionToImage(rxn)
    compressed_smiles = compress(smiles)
    file_location = f'{directory_location}/reactions/{compressed_smiles}.png'
    if xampp_installed and not os.path.exists(file_location):
        rimage.save(file_location)
    return f'http://localhost/reactions/{compressed_smiles}.png'


def get_file_location():
    return str(pathlib.Path(__file__).parent.absolute())


def get_fragments(smiles, fragments_df=None):
    smiles_mol = MolFromSmiles(smiles)
    if smiles_mol is None:
        return None

    if fragments_df is None:
        path = pathlib.Path(__file__).parent.absolute()
        fragments_df = pd.read_csv(str(path) + '/datafiles/Function-Groups-SMARTS.csv')

    frags = []
    for i in range(0, len(fragments_df)):
        row = dict(fragments_df.loc[i, :])
        if str(row['Sub-SMARTS']) != 'nan':
            sub_mol = MolFromSmarts(str(row['Sub-SMARTS']))
            main_mol = MolFromSmarts(str(row['SMARTS']))
            sub_mol_matches = len(smiles_mol.GetSubstructMatches(sub_mol))
            main_mol_matches = len(smiles_mol.GetSubstructMatches(main_mol))
            if sub_mol_matches == 0:
                pass
            elif sub_mol_matches == main_mol_matches:
                frags.append(row['Fragment'])
            elif sub_mol_matches > main_mol_matches > 0:
                frags.append(row['Fragment'])
                frags.append(row['Sub-Fragment'])
            else:
                frags.append(row['Sub-Fragment'])
        else:
            main_mol = MolFromSmarts(str(row['SMARTS']))
            main_mol_matches = len(smiles_mol.GetSubstructMatch(main_mol))
            if main_mol_matches > 0:
                frags.append(row['Fragment'])
    return list(set(frags))

import pandas as pd
from rdkit import Chem


def predict_pka(file):

    raw_data = pd.read_csv(file)
    smiles = list(raw_data['Smiles'])
    pka_data = pd.read_excel('sorting_data.xlsx')
    all_matches = []

    for smile in smiles:
        matches = []
        mol = Chem.MolFromSmiles(smile)
        for index, pka_predictor in pka_data.iterrows():
            pka_predictor_dict = dict(pka_predictor)
            if str(pka_predictor_dict['Conjugate Acid']) == 'nan':
                conj_base_mol = Chem.MolFromSmarts(pka_predictor_dict['Conjugate Base'])
                pka = pka_predictor_dict['pKa Value']
                if mol.HasSubstructMatch(conj_base_mol):
                    matches.append('({})={}'.format(pka_predictor_dict['Conjugate Base'], pka))
            elif str(pka_predictor_dict['Conjugate Base']) == 'nan':
                conj_acid_mol = Chem.MolFromSmarts(pka_predictor_dict['Conjugate Acid'])
                pka = pka_predictor_dict['pKa Value']
                if mol.HasSubstructMatch(conj_acid_mol):
                    matches.append('({})={}'.format(pka_predictor_dict['Conjugate Acid'], pka))
            else:
                conj_acid_mol = Chem.MolFromSmarts(pka_predictor_dict['Conjugate Acid'])
                conj_base_mol = Chem.MolFromSmarts(pka_predictor_dict['Conjugate Base'])
                pka = pka_predictor_dict['pKa Value']
                if mol.HasSubstructMatch(conj_acid_mol):
                    matches.append('({})={}'.format(pka_predictor_dict['Conjugate Acid'], pka))
                if mol.HasSubstructMatch(conj_base_mol):
                    matches.append('({})={}'.format(pka_predictor_dict['Conjugate Base'], pka))
        all_matches.append(matches)
    raw_data['pka'] = all_matches
    raw_data.to_csv(file, index=False)


predict_pka(r'C:\Users\User\PycharmProjects\McQuade-Chem-ML\Neo4j\BulkChem\BulkChemData.csv')
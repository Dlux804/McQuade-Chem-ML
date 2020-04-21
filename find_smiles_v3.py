import pandas as pd
import os
import xml.etree.ElementTree as ET
from core.misc import cd
from rdkit import Chem
import csv
import copy
"""
    Objective: Make CSVs from xml files and convert all the SMILES to canonical. If SMILES inthe list of SMILES provided 
    is in the CSVs, leave the extracted information in the CSV.  
"""

try:
    os.mkdir("./return_csv")
except OSError as e:
    pass

def split_dict(gen_dict):
    """
    Goal: Clean the dictionary provided into something more readable
    :param gen_dict:
    :return:
    """

    values = list(gen_dict.values())
    if len(values) < 2:
        return None
    if values[0][0:4] == 'cml:':
        values[0] = values[0][4:]
    clean_text = values[0] + ':' + values[1]
    return clean_text


def get_compound_info(compound):
    chemical_name_list = []
    identifiers_list = []
    amounts_list = []
    appearances_list = []

    molecule = compound.find('{http://www.xml-cml.org/schema}molecule')
    for child in molecule:
        chemical_name_list.append(child.text)
    identifiers = compound.findall('{http://www.xml-cml.org/schema}identifier')
    for identifier in identifiers:
        identifiers_list.append(split_dict(identifier.attrib))
    amounts = compound.findall('{http://www.xml-cml.org/schema}amount')
    for amount in amounts:
        amounts_list.append(amount.text)
    appearances = compound.findall('{http://bitbucket.org/dan2097}appearance')
    for appearance in appearances:
        appearances_list.append(appearance.text)
    try:
        full_string = identifiers_list[0]
        smiles = full_string[7:]
        mol = Chem.MolFromSmiles(smiles)
        new_smiles = Chem.MolToSmiles(mol)
        identifiers_list[0] = 'smiles:' + new_smiles
    except IndexError:
        pass

    chemical_dict = {'chemical_names': chemical_name_list, 'identifiers': identifiers_list,
                     'amounts': amounts_list, 'appearances': appearances_list}

    return chemical_dict

def get_root(foldername):
    with cd(foldername):

        root_list = []
        for root, dirs, files in os.walk(foldername):
            root_list.append(root)
        root_list.pop(0)
        print('All path to file:', root_list)
    return root_list


def xml_to_csv(root_list):
    files_dicts = {}
    for root in root_list:
        with cd(root):
            for root, dirs, files in os.walk(root):

                for f in files:
                    if f.endswith('.xml'):
                        print("working with:", f)
                        tree = ET.parse(f)
                        root = tree.getroot()

                        reaction_list_dicts = []
                        for reaction in root:
                            reaction_smiles_list = []
                            sources_list = []
                            reactants_list = []
                            products_list = []
                            solvents_list = []
                            catalyst_list = []
                            stages_list = []

                            reaction_smiles = reaction.findall('{http://bitbucket.org/dan2097}reactionSmiles')
                            for reaction_smile in reaction_smiles:
                                raw_reaction_smiles = reaction_smile.text
                                raw_reaction_smiles = raw_reaction_smiles.split('>')
                                for raw_reaction_smile in raw_reaction_smiles:
                                    reaction_smiles_list.append(raw_reaction_smile)

                            sources = reaction.find('{http://bitbucket.org/dan2097}source')
                            for source in sources:
                                sources_list.append(source.text)

                            reactantList = reaction.find('{http://www.xml-cml.org/schema}reactantList')
                            reactants = reactantList.findall("{http://www.xml-cml.org/schema}reactant")
                            for reactant in reactants:
                                final_reactant = get_compound_info(reactant)
                                reactants_list.append(final_reactant)

                            productList = reaction.find('{http://www.xml-cml.org/schema}productList')
                            products = productList.findall('{http://www.xml-cml.org/schema}product')
                            for product in products:
                                final_product = get_compound_info(product)
                                products_list.append(final_product)

                            spectatorList = reaction.find('{http://www.xml-cml.org/schema}spectatorList')
                            spectators = spectatorList.findall('{http://www.xml-cml.org/schema}spectator')
                            for spectator in spectators:
                                role = list(spectator.attrib.values())[0]
                                if role == 'solvent':
                                    final_solvent = get_compound_info(spectator)
                                    solvents_list.append(final_solvent)

                                if role == 'catalyst':
                                    final_catalyst = get_compound_info(spectator)
                                    catalyst_list.append(final_catalyst)

                            reactionActionList = reaction.find('{http://bitbucket.org/dan2097}reactionActionList')
                            reactionActions = reactionActionList.findall('{http://bitbucket.org/dan2097}reactionAction')
                            counter = 1
                            for reactionAction in reactionActions:
                                texts = reactionAction.findall('{http://bitbucket.org/dan2097}phraseText')
                                for text in texts:
                                    stages_list.append('step {}:'.format(counter) + text.text)
                                parameters = reactionAction.findall('{http://bitbucket.org/dan2097}parameter')
                                for parameter in parameters:
                                    if split_dict(parameter.attrib) is not None:
                                        stages_list.append(
                                            'step {} properties:'.format(counter) + split_dict(parameter.attrib))
                                counter = counter + 1

                            reaction_dict = {'reaction_smiles': reaction_smiles_list, 'sources': sources_list,
                                             'reactants': reactants_list, 'products': products_list,
                                             'solvents': solvents_list,
                                             'catalyst': catalyst_list, 'stages': stages_list}
                            reaction_list_dicts.append(reaction_dict)
                        files_dicts[f] = reaction_list_dicts
    return files_dicts


def find_smiles(files_dicts, smiles_list):
    print("Converting given SMILES to canonical")
    print()
    mol_list = list(map(Chem.MolFromSmiles, smiles_list))
    canon_smiles_list = list((map(Chem.MolToSmiles, mol_list)))
    print("Canonical form of given SMILES:", canon_smiles_list)
    with cd('return_csv/'):
        final_dict_list = []
        canon_name_list = []
        for canon in canon_smiles_list:
            main_key_list = []
            for main_key, value_dict in files_dicts.items():
                canon_name_list.append(canon)
                main_key_list.append(main_key)
                real_dicts_list = []
                for real_dicts in value_dict:
                    # print(real_dicts)
                    remove_dict = copy.deepcopy(real_dicts)
                    remove = ['reaction_smiles', 'sources', 'stages']
                    [remove_dict.pop(rem, None) for rem in remove]
                    for dict_k, list_v in remove_dict.items():
                        for final_dict in list_v:
                            try:
                                # print(final_dict)
                                all_smiles = final_dict['identifiers']
                                full_string = str(all_smiles[0])
                                if full_string[7:] == canon:
                                    real_dicts_list.append(real_dicts)

                            except IndexError:
                                pass
                final_dict_list.append(real_dicts_list)

        for canon_name, main_key_name, dict_df in zip(canon_name_list, main_key_list, final_dict_list):
            if len(dict_df) > 0:
                print("Found %s match in:" % canon_name, main_key_name)
                all_data = pd.DataFrame.from_records(dict_df)
                all_data.to_csv(main_key_name[:-4] + '_' + str(canon_name) + '.csv', index=False)
            else:
                print("No match for " + canon_name, main_key_name)
                pass


smiles_list = ['COC(=O)Cc1ccc(OC)cc1']
root_list = get_root('C:/Users/quang/McQuade-Chem-ML/xml')
files_dicts = xml_to_csv(root_list)
find_smiles(files_dicts, smiles_list)


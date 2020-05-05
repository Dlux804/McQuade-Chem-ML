import pandas as pd
import os
import xml.etree.ElementTree as ET
from core.misc import cd
from rdkit import Chem
import csv
import numpy as np
import copy
import memory_profiler
import time
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
    # clean_text = values[0] + ':' + values[1]
    clean_text = '{0}:{1}'.format(values[0], values[1])
    return clean_text


def get_compound_info(compound):
    molecule = compound.find('{http://www.xml-cml.org/schema}molecule')
    chemical_name_list = [child.text for child in molecule]
    # for child in molecule:
    #     chemical_name_list.append(child.text)
    identifiers = compound.findall('{http://www.xml-cml.org/schema}identifier')
    identifiers_list = [split_dict(identifier.attrib) for identifier in identifiers]
    # for identifier in identifiers:
    #     identifiers_list.append(split_dict(identifier.attrib))
    amounts = compound.findall('{http://www.xml-cml.org/schema}amount')
    amounts_list = [amount.text for amount in amounts]
    # for amount in amounts:
    #     amounts_list.append(amount.text)
    appearances = compound.findall('{http://bitbucket.org/dan2097}appearance')
    appearances_list = [appearance.text for appearance in appearances]
    # for appearance in appearances:
    #     appearances_list.append(appearance.text)
    try:
        full_string = identifiers_list[0]
        smiles = full_string[7:]
        # print("XML SMILES:", smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:

            pass
        else:
            new_smiles = Chem.MolToSmiles(mol)
            identifiers_list[0] = 'smiles:' + new_smiles
    except IndexError:
        pass
    # print('Error SMILES', error_smiles)
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
            for roots, dirs, files in os.walk(root):
                for f in files:
                    if f.endswith('.xml'):
                        print("working with:", f)
                        tree = ET.parse(f)
                        root = tree.getroot()

                        reaction_list_dicts = []
                        for reaction in root:
                            reaction_smiles_list = []
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
                            sources_list = [source.text for source in sources]


                            reactantList = reaction.find('{http://www.xml-cml.org/schema}reactantList')
                            reactants = reactantList.findall("{http://www.xml-cml.org/schema}reactant")
                            reactants_list = list(map(get_compound_info, reactants))


                            productList = reaction.find('{http://www.xml-cml.org/schema}productList')
                            products = productList.findall('{http://www.xml-cml.org/schema}product')
                            products_list = list(map(get_compound_info, products))
                            # for product in products:
                            #     final_product = get_compound_info(product)
                            #     products_list.append(final_product)

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
                                    stages_list.append('step {}:{}'.format(counter, text.text))
                                parameters = reactionAction.findall('{http://bitbucket.org/dan2097}parameter')
                                for parameter in parameters:
                                    if split_dict(parameter.attrib) is not None:

                                        stages_list.append('{0}{1}'.format('step {} properties:'.format(counter),
                                                                           split_dict(parameter.attrib)))
                                counter = counter + 1

                            reaction_dict = {'reaction_smiles': reaction_smiles_list, 'sources': sources_list,
                                             'reactants': reactants_list, 'products': products_list,
                                             'solvents': solvents_list,
                                             'catalyst': catalyst_list, 'stages': stages_list}
                            reaction_list_dicts.append(reaction_dict)
                        files_dicts[f] = reaction_list_dicts
    return files_dicts


def find_smiles(files_dicts, smiles_list):
    # print("Converting given SMILES to canonical")
    # print()
    mol_list = list(map(Chem.MolFromSmiles, smiles_list))
    canon_smiles_list = list((map(Chem.MolToSmiles, mol_list)))
    print("Canonical form of given SMILES:", canon_smiles_list)
    # length = len(dictfiles_dicts.keys())
    with cd('return_csv/'):
        for canon in canon_smiles_list:
            for main_key in files_dicts:
                real_dicts_list = []
                for real_dicts in files_dicts[main_key]:
                    # print(real_dicts)
                    remove_dict = copy.deepcopy(real_dicts)
                    remove = ['reaction_smiles', 'sources', 'stages']
                    [remove_dict.pop(rem, None) for rem in remove]
                    for dict_k in remove_dict:
                        for final_dict in remove_dict[dict_k]:
                            try:
                                if canon == final_dict['identifiers'][0][7:]:
                                    real_dicts_list.append(real_dicts)
                            except IndexError:
                                pass
                if len(real_dicts_list) > 0:
                    all_data = pd.DataFrame.from_records(real_dicts_list)
                    all_data.to_csv("{0}_{1}{2}".format(main_key[:-4], canon, '.csv'), index=False)
                else:
                    pass


smiles_list = ['C#C', '[C-]#[O+]', 'CCN(CC)CC', 'C1COCCO1', 'O=C1OCCC1', 'O=C([H])C', 'O=[C]OO[Na].[Na]', 'O=C(OCC)C',
               'O=C1OCCC1C(C)=O', '[H]C([H])=O', '[HH]', '[Cu]=O', 'O=[Bi][Bi](=O)=O', 'O=[Si]=O', 'OCCCCO', '[Cu]',
               '[Zn]=O', '[Al+3].[Al+3].[O-2].[O-2].[O-2]', '[Zr+4].[O-2].[O-2]', 'CC(C)=O', 'S=C=S', 'C=C=O',
               'C=C(C1)OC1=O', '[Na]OC', 'O1CC1', 'CO', 'CC(O)=O', 'O=P(OCC)(OCC)OCC', 'O=C1OCCC1', 'CCOC(C)=O',
               'O=C1OCC/C1=C(O[Na])\\C', 'O=C1OCCC1C(C)=O', 'O=C(CCCI)C', 'CCNCCO', 'CCN(CCCC(C)=O)CCO',
               'CCN(CCC/C(C)=N/O)CCO', 'CCN(CCCC(C)N)CCO', 'ClC1=CC=NC2=CC(Cl)=CC=C21',
               'ClC1=CC=C2C(N=CC=C2NC(CCCN(CC)CCO)C)=C1']








root_list = get_root('C:/Users/quang/McQuade-Chem-ML/xml')
files_dicts = xml_to_csv(root_list)

m1 = memory_profiler.memory_usage()
t1 = time.perf_counter()

find_smiles(files_dicts, smiles_list)

t2 = time.perf_counter()
m2 = memory_profiler.memory_usage()
time_diff = t2 - t1
mem_diff = m2[0] - m1[0]
print(f"It took {time_diff} Secs and {mem_diff} Mb to execute this method")


# data = ['Cl(=O)(=O)(=O)F']
# mol = Chem.MolFromSmiles('Cl(=O)(=O)(=O)F')
# if mol is None:
#     print(data)
# else:
#     smiles = Chem.MolToSmiles(mol)




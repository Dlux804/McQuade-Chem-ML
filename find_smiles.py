import pandas as pd
import os
import xml.etree.ElementTree as ET
from core.misc import cd
from rdkit import Chem

"""
    Objective: Make CSVs from xml files and convert all the SMILES to canonical. If SMILES inthe list of SMILES provided 
    is in the CSVs, leave the extracted information in the CSV.  
"""

smiles_list = ['CCCCCCCCCCCC', 'C=C(C)C(=O)OC']


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
    for root in root_list:
        with cd(root):
            for root, dirs, files in os.walk(root):
                for f in files:
                    if f.endswith('.xml'):
                        print("working with:", f)
                        tree = ET.parse(f)
                        root = tree.getroot()

                        reaction_dicts = []
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
                            final_length = len(reactants_list+products_list+catalyst_list+solvents_list)
                            if final_length > 0:
                                reaction_dicts.append(reaction_dict)
                            else:
                                pass
                        all_data = pd.DataFrame.from_records(reaction_dicts)
                        all_data.to_csv(f[:-4] + '.csv', index=False)
                        #

root_list = get_root('C:/Users/quang/McQuade-Chem-ML/xml')
xml_to_csv(root_list)
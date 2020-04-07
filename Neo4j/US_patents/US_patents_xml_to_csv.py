import pandas as pd
import os
import xml.etree.ElementTree as ET


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

    chemical_dict = {'chemical_names': chemical_name_list, 'identifiers': identifiers_list,
                     'amounts': amounts_list, 'appearances': appearances_list}
    return chemical_dict


def split_dict(gen_dict):
    values = list(gen_dict.values())
    if len(values) < 2:
        return None
    if values[0][0:4] == 'cml:':
        values[0] = values[0][4:]
    clean_text = values[0] + ' = ' + values[1]
    return clean_text


def xml_to_csv(input_file, output_file):
    tree = ET.parse(input_file)
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
            reactants_list.append(get_compound_info(reactant))

        productList = reaction.find('{http://www.xml-cml.org/schema}productList')
        products = productList.findall('{http://www.xml-cml.org/schema}product')
        for product in products:
            products_list.append(get_compound_info(product))

        spectatorList = reaction.find('{http://www.xml-cml.org/schema}spectatorList')
        spectators = spectatorList.findall('{http://www.xml-cml.org/schema}spectator')
        for spectator in spectators:
            role = list(spectator.attrib.values())[0]
            if role == 'solvent':
                solvents_list.append(get_compound_info(spectator))
            if role == 'catalyst':
                catalyst_list.append(get_compound_info(spectator))

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
                    stages_list.append('step {} properties:'.format(counter) + split_dict(parameter.attrib))
            counter = counter + 1

        reaction_dict = {'reaction_smiles': reaction_smiles_list, 'sources': sources_list,
                         'reactants': reactants_list, 'products': products_list, 'solvents': solvents_list,
                         'catalyst': catalyst_list, 'stages': stages_list}
        reaction_dicts.append(reaction_dict)

    all_data = pd.DataFrame.from_records(reaction_dicts)
    all_data.to_csv(output_file, index=False)


def US_grants_directory_to_csvs(path_to_directory):
    main_directories = os.listdir(path_to_directory)
    for main_directory in main_directories:
        main_directory = path_to_directory + "/" + main_directory
        for directory in os.listdir(main_directory):
            if len(directory.split('_')) == 1:
                print(directory)
                output_directory = directory + '_csv'
                output_directory = main_directory + "/" + output_directory
                if not os.path.exists(output_directory):
                    os.mkdir(output_directory)
                directory = main_directory + "/" + directory
                for file in os.listdir(directory):
                    input_file = directory + '/' + file
                    csv_file = file.split('.')[0]
                    csv_file = str(csv_file) + '.csv'
                    output_file = output_directory + '/' + csv_file
                    try:
                        xml_to_csv(input_file, output_file)
                    except:
                        pass


def clean_up_checker_files(path_to_directory):
    main_directories = os.listdir(path_to_directory)
    for main_directory in main_directories:
        main_directory = path_to_directory + "/" + main_directory
        for directory in os.listdir(main_directory):
            directory = main_directory + '/' + directory
            for file in os.listdir(directory):
                file = directory + '/' + file
                split_file = file.split('.')
                if split_file[len(split_file)-1] == 'checker':
                    os.remove(file)
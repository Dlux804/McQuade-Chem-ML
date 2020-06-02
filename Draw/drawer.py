import os
import pathlib
import shutil
import hashlib

from PIL import Image
from rdkit.Chem import Draw, MolFromSmarts, MolFromSmiles, MolToSmiles, rdChemReactions

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


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


def save_rdkit_reaction_image(rxn_smiles, directory_location, svg=False):
    if not os.path.exists(directory_location):
        raise Exception('Directory not found')

    compressed_smiles = compress(rxn_smiles)
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smiles)

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


def __dc__(compounds, typ):
    counter = 0
    path = str(pathlib.Path(__file__).parent.absolute())
    compounds = compounds.split('.')
    for compound in compounds:
        compound = MolFromSmarts(compound)
        compound = MolFromSmiles(MolToSmiles(compound))
        if compound is not None:
            [x.ClearProp('molAtomMapNumber') for x in compound.GetAtoms()]
            Draw.MolToFile(compound, f'{path}/temp_files/{typ}/{counter}.png')
            counter = counter + 1


def save_reaction_image(rxn_smiles, location):

    path = str(pathlib.Path(__file__).parent.absolute())
    if os.path.exists(f'{path}/temp_files'):
        shutil.rmtree(f'{path}/temp_files')

    os.mkdir(f'{path}/temp_files')
    os.mkdir(f'{path}/temp_files/R')
    os.mkdir(f'{path}/temp_files/P')
    os.mkdir(f'{path}/temp_files/A')
    plus_sign = f'{path}/plus.png'
    reaction_arrow = f'{path}/reaction_arrow.jpeg'

    reactants = rxn_smiles.split('>')[0]
    agents = rxn_smiles.split('>')[1]
    products = rxn_smiles.split('>')[2]

    __dc__(reactants, 'R')
    __dc__(products, 'P')
    __dc__(agents, 'A')

    for directory in os.listdir(f'{path}/temp_files'):
        typ = directory
        directory = f'{path}/temp_files/{directory}'
        new_files = []
        for file in os.listdir(directory):
            file = f'{directory}/{file}'
            new_files.append(file)
            new_files.append(plus_sign)

        if len(new_files) > 0:
            files = new_files
            files.pop(len(files)-1)

            images = [Image.open(x) for x in files]
            widths, heights = zip(*(i.size for i in images))

            total_width = sum(widths)
            max_height = max(heights)

            new_im = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            new_im.save(f'{path}/temp_files/{typ}.png')
            shutil.rmtree(f'{directory}')

        else:
            shutil.rmtree(f'{directory}')

    if not os.path.exists(f'{path}/temp_files/A.png'):
        reaction_arrow_image = Image.open(reaction_arrow)
        arrow_width, arrow_height = reaction_arrow_image.size
        new_im = Image.new('RGB', (arrow_width, arrow_height ))
        new_im.paste(reaction_arrow_image)
        new_im_width, new_im_height = new_im.size

        ratio = 300 / new_im_height
        new_im = new_im.resize((int(new_im_width * ratio), int(new_im_height * ratio)), Image.ANTIALIAS)

        new_im.save(f'{path}/temp_files/A.png')

    else:

        reaction_arrow_image = Image.open(reaction_arrow)
        arrow_width, arrow_height = reaction_arrow_image.size

        a_image = Image.open(f'{path}/temp_files/A.png')
        a_width, a_height = a_image.size
        ratio = arrow_width / a_width
        a_image = a_image.resize((int(a_width * ratio), int(a_height * ratio)), Image.ANTIALIAS)
        a_width, a_height = a_image.size

        total_height = arrow_height + a_height
        max_width = max(arrow_width, a_width)

        new_im = Image.new('RGB', (max_width, total_height))

        new_im.paste(reaction_arrow_image, (0, a_height))
        new_im.paste(a_image, (0, 0))
        new_im_width, new_im_height = new_im.size

        ratio = 300 / new_im_height
        new_im = new_im.resize((int(new_im_width * ratio), int(new_im_height * ratio)), Image.ANTIALIAS)

        new_im.save(f'{path}/temp_files/A.png')

    files = [f'{path}/temp_files/R.png', f'{path}/temp_files/A.png', f'{path}/temp_files/P.png']

    images = [Image.open(x) for x in files]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save(location)
    shutil.rmtree(f'{path}/temp_files')

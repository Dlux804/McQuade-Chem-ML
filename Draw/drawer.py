import os
import pathlib
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


def __dc__(compounds, path):
    counter = 0
    images = []
    plus_sign = f'{path}/plus.png'
    compounds = compounds.split('.')
    for compound in compounds:
        compound = MolFromSmarts(compound)
        compound = MolFromSmiles(MolToSmiles(compound))
        if compound is not None:
            [x.ClearProp('molAtomMapNumber') for x in compound.GetAtoms()]
            images.append(Draw.MolToImage(compound))
            images.append(Image.open(plus_sign))
            counter = counter + 1
    images.pop(len(images)-1)
    return images


def __dcs__(compounds):
    if len(compounds) > 0:
        widths, heights = zip(*(i.size for i in compounds))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in compounds:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        return new_im


def __cga__(agents, path):
    reaction_arrow = f'{path}/reaction_arrow.jpeg'
    if not agents:
        reaction_arrow_image = Image.open(reaction_arrow)
        arrow_width, arrow_height = reaction_arrow_image.size
        new_im = Image.new('RGB', (arrow_width, arrow_height))
        new_im.paste(reaction_arrow_image)
        new_im_width, new_im_height = new_im.size

        ratio = 300 / new_im_height
        new_im = new_im.resize((int(new_im_width * ratio), int(new_im_height * ratio)), Image.ANTIALIAS)

        return new_im

    else:

        reaction_arrow_image = Image.open(reaction_arrow)
        arrow_width, arrow_height = reaction_arrow_image.size

        a_image = agents
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

        return new_im


def save_reaction_image(rxn_smiles, location):
    path = str(pathlib.Path(__file__).parent.absolute())

    reactants = rxn_smiles.split('>')[0]
    agents = rxn_smiles.split('>')[1]
    products = rxn_smiles.split('>')[2]

    reactants = __dcs__(__dc__(reactants, path))
    products = __dcs__(__dc__(products, path))
    agents = __dcs__(__dc__(agents, path))
    agents = __cga__(agents, path)

    images = [reactants, agents, products]
    new_im = __dcs__(images)
    new_im.save(location)

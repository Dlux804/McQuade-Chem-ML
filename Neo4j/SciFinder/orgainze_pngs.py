import cv2
from PIL import Image
import os
import numpy as np
from shutil import copy

def same_image(imageA, imageB):
    def resize_image(image):
        image_file = image.split('/')
        image_file = image_file[len(image_file) - 1]
        image_file = "COMPRE$$ED" + image_file
        size = (64, 64)
        image = Image.open(image)
        image.thumbnail(size, Image.ANTIALIAS)
        background = Image.new('RGBA', size, (255, 255, 255, 0))
        background.paste(
            image, (int((size[0] - image.size[0]) / 2), int((size[1] - image.size[1]) / 2))
        )
        background.save(image_file)
        return image_file

    imageA = resize_image(imageA)
    cv_imageA = cv2.imread(imageA)
    imageB = resize_image(imageB)
    cv_imageB = cv2.imread(imageB)
    difference = np.sum(cv2.subtract(cv_imageA, cv_imageB)) / 1296
    if difference < 0.05:
        return True
    return False

def search_for_match(input_file):
    for compound_dir in os.listdir('Raw_Scifinder_Files/printed_scifinder'):
        compound_dir = 'Raw_Scifinder_Files/printed_scifinder/' + compound_dir
        for file in os.listdir(compound_dir):
            raw_file = file
            file = compound_dir + '/' + file
            if same_image(input_file, file):
                os.remove(input_file)
                return file
    return input_file

for reaction_dir in os.listdir('reactions'):
    print(reaction_dir)
    reaction_dir = 'reactions/' + reaction_dir
    reactants = os.listdir(reaction_dir + '/reactants')
    products = os.listdir(reaction_dir + '/products')
    for reactant in reactants:
        raw_reactant = reactant
        reactant = reaction_dir + '/reactants/' + reactant
        reactant = search_for_match(reactant)
        copy(reactant, reaction_dir + '/reactants/')
    for product in products:
        raw_product = product
        product = reaction_dir + '/products/' + product
        product = search_for_match(product)
        copy(product, reaction_dir + '/products')
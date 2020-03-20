import cv2
from PIL import Image
import os
import numpy as np
from shutil import copyfile


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
    os.remove(imageA)
    os.remove(imageB)
    if difference < 0.05:
        return True
    return False


def is_arrow(image):
    return same_image(image, 'Raw_Scifinder_Files/general_Files/arrow.png')


def is_plus(image):
    return same_image(image, 'Raw_Scifinder_Files/general_Files/plus.png')


def insert_products_and_reactants(reactants, products, counter, raw_compound_dir):
    cwd = os.getcwd()
    reaction = 'reaction_{0}'.format(counter)
    try:
        os.mkdir('reactions/' + '{0}_'.format(raw_compound_dir) + reaction)
        os.mkdir('reactions/' + '{0}_'.format(raw_compound_dir) + reaction + '/reactants')
        os.mkdir('reactions/' + '{0}_'.format(raw_compound_dir) + reaction + '/products')
    except PermissionError:
        pass
    except FileExistsError:
        pass
    for reactant in reactants:
        raw_reactant = reactant
        reactant = reactant.split('/')
        reactant = reactant[len(reactant)-1]
        copyfile(raw_reactant, 'reactions/' + '{0}_'.format(raw_compound_dir) + reaction + '/reactants/' + reactant)
    for product in products:
        raw_product = product
        product = product.split('/')
        product = product[len(product)-1]
        copyfile(raw_product, 'reactions/' + '{0}_'.format(raw_compound_dir) + reaction + '/products/' + product)
    os.chdir(cwd)


compound_dirs = os.listdir('Raw_Scifinder_Files/printed_scifinder')
for compound_dir in compound_dirs:
    raw_compound_dir = compound_dir
    counter = 1
    compound_dir = 'Raw_Scifinder_Files/printed_scifinder/{0}'.format(compound_dir)
    reactants = []
    products = []
    is_reactants = True
    first_compound = True
    files = os.listdir(compound_dir)
    i = 0
    while i < len(files)-1:

        def fetch_file(i):
            fetched_file = files[i]
            return compound_dir + '/' + fetched_file

        def check_for_odd_reactant(i):
            if i >= len(files)-1:
                return False
            if not is_plus(fetch_file(i - 1)) and not is_plus(fetch_file(i + 1)):
                if not is_arrow(fetch_file(i - 1)) and not is_arrow(fetch_file(i + 1)):
                    if not is_arrow(file) and not is_plus(file):
                        return True
            return False

        while is_reactants:
            if i > len(files) - 1:
                break
            file = fetch_file(i)
            if file.split('.')[1] == 'ppm':
                i = i + 1
            elif check_for_odd_reactant(i):
                reactants.append(file)
                i = i + 1
                while True:
                    if check_for_odd_reactant(i):
                        reactants.append(fetch_file(i))
                        i = i + 1
                    else:
                        break
                first_compound = False
                reactants.append(fetch_file(i))
                i = i + 1
            elif first_compound:
                reactants.append(file)
                first_compound = False
                i = i + 1
            elif is_plus(file):
                i = i + 1
                file = fetch_file(i)
                reactants.append(file)
                i = i + 1
            else:
                first_compound = True
                is_reactants = False
                i = i + 1

        while not is_reactants:
            if i > len(files) - 1:
                break
            file = fetch_file(i)
            if file.split('.')[1] == 'ppm':
                i = i + 1
            elif first_compound:
                products.append(file)
                first_compound = False
                i = i + 1
            elif is_plus(file):
                i = i + 1
                file = fetch_file(i)
                products.append(file)
                i = i + 1
            else:
                first_compound = True
                is_reactants = True
                insert_products_and_reactants(reactants, products, counter, raw_compound_dir)
                counter = counter + 1
                reactants = []
                products = []

import pandas as pd


def get_review(product):
    try:
        return list(pd.read_csv(r'Katty-Files-v2\outputFiles\{}.csv'.format(product)))[0]
    except FileNotFoundError:
        return None
    except:
        raise Exception("Error from clean_kaitlins_data.py -> get_review")


def condense_data():
    files = ["50", "60", "70", "80", "90"]
    for file in files:
        raw_data = pd.read_excel(r'Katty-Files-v2\raw_files\prodIngrSMI{}.xlsx'.format(file))
        clean_data = []
        for i in range(len(raw_data)):
            inter_data = {}
            row = dict(raw_data.iloc[i])
            row = list(row.values())
            product_number = row[0]
            smiles = []
            if get_review(product_number):
                review = get_review(product_number)
                for x in range(len(row)):
                    if x == 0:
                        pass
                    else:
                        if str(row[x]) != 'nan':
                            smiles.append(row[x])
                inter_data['product_number'] = product_number
                inter_data['ingredients_smiles'] = smiles
                inter_data['review'] = review
                clean_data.append(inter_data)
        clean_data = pd.DataFrame(clean_data)
        clean_data.to_csv(r'Katty-Files-v2\raw_files\prodIngrSMI{}.csv'.format(file), index=False)


def combine_files():
    files = ["50", "60", "70", "80", "90"]
    all_data = []
    for file in files:
        if len(all_data) <= 0:
            all_data = pd.read_csv(r'Katty-Files-v2\raw_files\prodIngrSMI{}.csv'.format(file))
        else:
            test = pd.read_csv(r'Katty-Files-v2\raw_files\prodIngrSMI{}.csv'.format(file))
            all_data = all_data.append(test)
    all_data.to_csv('kat_data.csv', index=False)


combine_files()
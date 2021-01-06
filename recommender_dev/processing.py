import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def get_target_value(dataset, target, smiles):
    dataset_df = pd.read_csv(dataset)  # Get Lipo value from dataset
    value = dataset_df.loc[dataset_df['smiles'] == smiles].to_dict('records')[0][target]
    return value


def process_control_smiles(dataset, target, algo, results_dir):
    rows = []
    for directory in os.listdir(results_dir):  # Each directory is a single run and has only one control smiles
        directory = f"{results_dir}/{directory}"
        if os.path.isdir(directory):  # Results files get saved to same directory
            control_smiles = list(pd.read_csv(f"{directory}/control_smiles.csv")['smiles'])[0]
            row_data = {'control_smiles': control_smiles}
            row_data[target] = get_target_value(dataset, target, control_smiles)

            def __galc__():  # get average list score
                df = pd.read_csv(f"{directory}/ratio.csv")
                df = df.loc[df['type'] == algo]
                list_scores = df['ratio_score'].tolist()
                average_list_score = sum(list_scores) / len(list_scores)
                return average_list_score

            def __gagc__():  # get average goodness score
                control_df = pd.read_csv(f"{directory}/control_smiles.csv")
                best_pred_error = control_df.to_dict('records')[0]['pred_error']
                df = pd.read_csv(f"{directory}/{algo}_smiles.csv")
                unqiue_smiles = list(set(df['smiles'].tolist()))  # Only get unqiue smiles
                goodness_scores = []
                for smiles in unqiue_smiles:
                    sub_df = df.loc[df['smiles'] == smiles]
                    testing_bm = sub_df.to_dict('records')[0]['run_name']
                    goodness_score = control_df.loc[control_df['run_name'] == testing_bm]
                    goodness_score = goodness_score.to_dict('records')[0]['pred_error']
                    goodness_score = goodness_score - best_pred_error
                    goodness_scores.append(goodness_score)
                average_goodness_score = sum(goodness_scores) / len(goodness_scores)
                return average_goodness_score

            def __gpr__():  # get percent right
                control_df = pd.read_csv(f"{directory}/control_smiles.csv")
                bm = control_df.to_dict('records')[0]['run_name']  # bm = best model (ground turth)
                df = pd.read_csv(f"{directory}/{algo}_smiles.csv")
                unqiue_smiles = list(set(df['smiles'].tolist()))
                top_list = []
                for smiles in unqiue_smiles:
                    sub_df = df.loc[df['smiles'] == smiles]
                    testing_bm = sub_df.to_dict('records')[0]['run_name']
                    if testing_bm == bm:
                        top_list.append(1)
                    else:
                        top_list.append(0)
                percent_right = sum(top_list) / len(top_list)
                return percent_right

            def __gptt__():  # get percent top three
                control_df = pd.read_csv(f"{directory}/control_smiles.csv")
                bm = control_df.to_dict('records')[0]['run_name']
                df = pd.read_csv(f"{directory}/{algo}_smiles.csv")
                unqiue_smiles = list(set(df['smiles'].tolist()))
                top_list = []
                for smiles in unqiue_smiles:
                    sub_df = df.loc[df['smiles'] == smiles]
                    testing_bms = sub_df['run_name'].tolist()[:3]  # top three models
                    if bm in testing_bms:
                        top_list.append(1)
                    else:
                        top_list.append(0)
                percent_right = sum(top_list) / len(top_list)
                return percent_right

            row_data['list_score'] = __galc__()
            row_data['goodness_score'] = __gagc__()
            row_data['percent_right'] = __gpr__()
            row_data['percent_top_3'] = __gptt__()
            rows.append(row_data)

    algo_df = pd.DataFrame(rows)
    return algo_df


def process_sim_smiles(dataset, target, algo, results_dir):
    rows = []
    for directory in os.listdir(results_dir):  # Each directory is a single run and has only one control smiles
        directory = f"{results_dir}/{directory}"
        if os.path.isdir(directory):  # Results files get saved to same directory
            algo_df = pd.read_csv(f"{directory}/{algo}_smiles.csv")
            unqiue_smiles = list(set(algo_df['smiles'].tolist()))

            def __gsc__(smi):  # get sim score
                df = pd.read_csv(f"{directory}/ratio.csv")
                df = df.loc[df['type'] == algo]
                df = df.loc[df['smiles'] == smi]
                list_score = df.to_dict('records')[0][f'{algo}_similarity']
                return list_score

            def __glc__(smi):  # get list score
                df = pd.read_csv(f"{directory}/ratio.csv")
                df = df.loc[df['type'] == algo]
                df = df.loc[df['smiles'] == smi]
                list_score = df.to_dict('records')[0]['ratio_score']
                return list_score

            def __ggc__(smi):  # get goodness score
                control_df = pd.read_csv(f"{directory}/control_smiles.csv")
                best_pred_error = control_df.to_dict('records')[0]['pred_error']
                bm = control_df.to_dict('records')[0]['run_name']  # bm = best model (ground turth)
                df = pd.read_csv(f"{directory}/{algo}_smiles.csv")
                sub_df = df.loc[df['smiles'] == smi]
                testing_bm = sub_df.to_dict('records')[0]['run_name']
                goodness_score = control_df.loc[control_df['run_name'] == testing_bm]
                goodness_score = goodness_score.to_dict('records')[0]['pred_error']
                goodness_score = goodness_score - best_pred_error
                return goodness_score

            for smiles in unqiue_smiles:
                row = {'test_smiles': smiles, target: get_target_value(dataset, target, smiles),
                       'sim score': __gsc__(smiles), 'list score': __glc__(smiles),
                       'goodness score': __ggc__(smiles)}
                rows.append(row)

    algo_df = pd.DataFrame(rows)
    return algo_df


def plot(algo, df, x_name, y_name, color=None):
    x = df[x_name]
    y = df[y_name]

    plt.rcParams['figure.figsize'] = [12, 9]
    plt.style.use('bmh')
    fig, ax = plt.subplots()
    if color is not None:
        norm = cm.colors.Normalize(vmax=df[color].max(), vmin=df[color].min())
        plt.scatter(x, y, c=df[color], cmap='plasma', norm=norm, alpha=0.7)
        cbar = plt.colorbar()
        cbar.set_label(color)
    else:
        plt.scatter(x, y, alpha=0.7)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    plt.title(f'{x_name} vs {y_name} ({algo})')
    fig.patch.set_facecolor('blue')  # Will change background color
    fig.patch.set_alpha(0.0)  # Makes background transparent
    # plt.show()
    plt.close()


def plot_target_vs_list_score_control(target, algo, cs_df):  # cs = control smiles
    plot(algo, cs_df, target, 'list_score', color=None)


if __name__ == "__main__":
    results_dir = 'results'
    dataset = "recommender_test_files/lipo_raw.csv"
    target = 'exp'

    algos = ['rdkit', 'hyer', 'jaccard']
    for algo in algos:
        df_1 = process_control_smiles(dataset, target, algo, results_dir)
        plot_target_vs_list_score_control(target, algo, df_1)
        df_2 = process_sim_smiles(dataset, target, algo, results_dir)

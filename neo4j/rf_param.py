import pandas as pd
from py2neo import Graph, Node, Relationship
import cypher  # Run cypher commands
import extract_params as ep  # Extract parameters from ml results
import make_labels as ml  # Make label dataframes


def graph_rfparam(model_csv, algor="rf"):
    """

    :param model_csv:
    :param algor:
    :return:
    """

    graph = Graph("bolt://localhost:7687", user="neo4j", password="1234")
    model_df = pd.read_csv(model_csv)  # csv file with all ml results
    label_modeldf = ml.label_model_todf(model_csv)
    full_modeldf = pd.concat([model_df, label_modeldf], axis=1)
    algo_modeldf = full_modeldf[full_modeldf.algorithm == algor]  # dataframe with specific algorithms
    model_dicts = algo_modeldf.to_dict('records')
    # print(model_dicts)
    print("Full model dataframe\n")
    print(algo_modeldf)
    param_df = ep.param_finaldf(model_csv, algor)
    label_paramdf = ml.label_param_todf(model_csv, algor)
    full_labeldf = pd.concat([param_df, label_paramdf], axis=1)
    algo_paramdf = full_labeldf[full_labeldf.algorithm == algor]
    print("Full param datafrmae\n")
    print(algo_paramdf)
    # print(algo_paramdf.columns.tolist())
    param_dct = algo_paramdf.to_dict('records')  # Dict of dataframe for ml parameters
    # print(param_dct)
    for i in range(len(model_dicts)):
        ml_dict = model_dicts[i]
        tx = graph.begin()
        print('Creating model nodes number: ' + str(i))
        runs = Node("run_num", run=ml_dict['Run'])
        tx.create(runs)
        algo = Node("algo", algorithm=ml_dict['algorithm'])
        tx.create(algo)
        data = Node("data_ml", data=ml_dict['dataset'])
        tx.create(data)
        target = Node("targets", target=ml_dict['target'])
        tx.create(target)
        feat_meth = Node("featmeth", feat_meth=ml_dict['feat_meth'])
        tx.create(feat_meth)
        feat_time = Node("feattime", feat_time=ml_dict['feat_time'])
        tx.create(feat_time)
        tuned = Node("tuned", tune=ml_dict['tuned'])
        tx.create(tuned)
        feature_list = Node("featurelist", feature_lists=ml_dict['feature_list'])
        tx.create(feature_list)
        regressor = Node("regress", regressor=ml_dict['regressor'])
        tx.create(regressor)
        tunetime = Node("tunetimes", tunetime=ml_dict['tuneTime'])
        tx.create(tunetime)
        r2_avg_label = Node("r2avg#", r2_avg_labels=ml_dict['r2_avgRun#'])
        tx.create(r2_avg_label)
        r2_avg = Node("r2avg", r2_avg=ml_dict['r2_avg'])
        tx.create(r2_avg)
        r2_std_label = Node("r2std#", r2_std_label=ml_dict['r2_stdRun#'])
        tx.create(r2_std_label)
        r2_std = Node("r2std", r2_std=ml_dict['r2_std'])
        tx.create(r2_std)
        mse_avg_label = Node("mseavg#", mse_avg_label=ml_dict['mse_avgRun#'])
        tx.create(mse_avg_label)
        mse_avg = Node("mseavg", mse_avg=ml_dict['mse_avg'])
        tx.create(mse_avg)
        mse_std_label = Node("msestd#", mse_std_label=ml_dict['mse_stdRun#'])
        tx.create(mse_std_label)
        mse_std = Node("msestd", mse_std=ml_dict['mse_std'])
        tx.create(mse_std)
        rmse_avg_label = Node("rmseavg#", rmse_avg_label=ml_dict['rmse_avgRun#'])
        tx.create(rmse_avg_label)
        rmse_avg = Node("rmseavg", rmse_avg=ml_dict['rmse_avg'])
        tx.create(rmse_avg)
        rmse_std_label = Node("rmsestd#", rmse_std_label=ml_dict['rmse_stdRun#'])
        tx.create(rmse_std_label)
        rmse_std = Node("rmsestd", rmse_std=ml_dict['rmse_std'])
        tx.create(rmse_std)
        time_avg = Node("timeavg", time_avg=ml_dict['time_avg'])
        tx.create(time_avg)
        time_std = Node("timestd", time_std=ml_dict['time_std'])
        tx.create(time_std)
        final_results = Node("results", result=ml_dict['Results'])
        tx.create(final_results)
        print('Creating Relationships Number ' + str(i))
        aa = Relationship(runs, "uses", algo)
        tx.merge(aa)
        ab = Relationship(runs, "uses", data)
        tx.merge(ab)
        ac = Relationship(data, "has", target)
        tx.merge(ac)
        ad = Relationship(runs, "generates", feat_meth)
        tx.merge(ad)
        ae = Relationship(feature_list, "feat_time", feat_time)
        tx.merge(ae)
        af = Relationship(feat_meth, "means", feature_list)
        tx.merge(af)
        ag = Relationship(tuned, "tuned", regressor)
        tx.merge(ag)
        ah = Relationship(algo, "params", regressor)
        tx.merge(ah)
        ai = Relationship(tuned, "tunetime", tunetime)
        tx.merge(ai)
        aj = Relationship(regressor, "gives", final_results)
        tx.merge(aj)
        ak = Relationship(final_results, "has", r2_avg_label)
        tx.merge(ak)
        al = Relationship(r2_avg_label, "is", r2_avg)
        tx.merge(al)
        am = Relationship(final_results, "has", r2_std_label)
        tx.merge(am)
        an = Relationship(r2_std_label, "is", r2_std)
        tx.merge(an)
        ao = Relationship(final_results, "has", mse_avg_label)
        tx.merge(ao)
        ap = Relationship(mse_avg_label, "is", mse_avg)
        tx.merge(ap)
        aq = Relationship(final_results, "has", mse_std_label)
        tx.merge(aq)
        ar = Relationship(mse_std_label, "is", mse_std)
        tx.merge(ar)
        at = Relationship(final_results, "has", rmse_avg_label)
        tx.merge(at)
        au = Relationship(rmse_avg_label, "is", rmse_avg)
        tx.merge(au)
        av = Relationship(final_results, "has", rmse_std_label)
        tx.merge(av)
        aw = Relationship(rmse_std_label, "is", rmse_std)
        tx.merge(aw)
        az = Relationship(algo, "tune", tuned)
        tx.merge(az)
        bb = Relationship(runs, "gives", final_results)
        tx.merge(bb)
        bc = Relationship(algo, "contributes to", final_results)
        tx.merge(bc)
        bd = Relationship(data, "contributes to", final_results)
        tx.merge(bd)
        be = Relationship(feat_meth, "contributes to", final_results)
        tx.merge(be)
        loop_param = param_dct[i]
        bootstrap_label = Node("bootstrap_label", bootstrap_labels=loop_param['bootstrapRun#'])
        tx.create(bootstrap_label)
        bootstrap = Node("bootstrap", bootstraps=loop_param['bootstrap'])
        tx.create(bootstrap)
        maxdepth_label = Node("maxdepth_label", maxdepth_labels=loop_param['max_depthRun#'])
        tx.create(maxdepth_label)
        maxdepth = Node("maxdepth", maxdepths=loop_param['max_depth'])
        tx.create(maxdepth)
        max_features_label = Node("max_features_label", max_features_labels=loop_param['max_featuresRun#'])
        tx.create(max_features_label)
        max_features = Node("max_features", max_feature=loop_param['max_features'])
        tx.create(max_features)
        n_estimators_label = Node("nestimators_label", nestimators_label=loop_param['n_estimatorsRun#'])
        tx.create(n_estimators_label)
        n_estimators = Node("n_estimators", estimators=loop_param['n_estimators'])
        tx.create(n_estimators)
        bf = Relationship(regressor, "has", bootstrap_label)
        tx.merge(bf)
        bg = Relationship(bootstrap_label, "is", bootstrap)
        tx.merge(bg)
        bh = Relationship(regressor, "has", maxdepth_label)
        tx.merge(bh)
        bi = Relationship(maxdepth_label, "is", maxdepth)
        tx.merge(bi)
        bj = Relationship(regressor, "has", max_features_label)
        tx.merge(bj)
        bk = Relationship(max_features_label, "is", max_features)
        tx.merge(bk)
        bl = Relationship(regressor, "has", n_estimators_label)
        tx.merge(bl)
        bm = Relationship(n_estimators_label, "is", n_estimators)
        tx.merge(bm)
        tx.commit()


# graph_rfparam('ml_results3.csv')
# gdb_param.graph_gdbparam('ml_results3.csv')
# cypher.run_cypher_command(pd.read_csv('ml_results3.csv'), "target")
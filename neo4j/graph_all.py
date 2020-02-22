import gdb_param
import rf_param
import ada_param
import knn_param
import cypher
import pandas as pd




df = pd.read_csv('ml_results3.csv')

gdb_param.graph_gdbparam('ml_results3.csv')
rf_param.graph_rfparam('ml_results3.csv')
knn_param.graph_knnparam('ml_results3.csv')
ada_param.graph_adaparam('ml_results3.csv')


cypher.run_cypher_command(df, "algorithm")
cypher.run_cypher_command(df, "dataset")
cypher.run_cypher_command(df, "feat_meth")




from graph import ml_graph
from graph import cypher


def main(csv):
    """

    :param csv:
    :return:
    """
    graph = ml_graph.MlGraph(csv)
    df = graph.graph_model()
    graph.rf()
    graph.ada()
    graph.gdb()
    graph.knn()
    return df


if __name__ == '__main__':
    df = main('ml_results3.csv')
    command = cypher.Cypher()
    command.cypher_command(df, "regressor")
    command.cypher_command(df, "algorithm")
    command.cypher_command(df, "dataset")

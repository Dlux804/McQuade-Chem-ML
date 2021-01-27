from py2neo import Graph, ClientError
import pandas as pd
import os
from core.storage import cd
from py2neo import *
import pathlib
from main import ROOT_DIR
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
from core import name


def check_for_results_folder(results_directory):
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)


class CypherAutomate:

    def __init__(self, molecules_per_batch=1000, dataset=None):
        """"""
        self.dataset = dataset
        self.df = pd.read_csv(self.dataset)
        self.graph = None
        self.batch = molecules_per_batch

    def connect_to_neo4j(self, port="bolt://localhost:7687", username="neo4j", password="password"):
        """
        Objective: Check Neo4j connection
        :param port: Port to connect to Neo4j (can be http or bolt)
        :param username: Username for Neo4j database
        :param password: Password for Neo4j database
        :return:
        """

        self.graph = Graph(port, username=username, password=password)

    def check_for_dataset(self):
        """
        Objective: Check to see if the desired dataset is currently in the user's database
        :return:
        """
        query_results = self.graph.run(
            """
            MATCH (D:DataSet{data: $data})
            RETURN D.data
            """, parameters={'data': self.dataset}).data()
        if len(query_results) == 0:
            raise Exception(self.dataset + """ is not available in your current database.""")

    def countFrag(self):
        """
        Objective: Get the total number of fragments for all the molecules in a specific dataset
        :return:
        """
        query_results = self.graph.run(
            """
            MATCH (D:DataSet{data: $data})-[:CONTAINS_MOLECULE]->(M:Molecule)-[f:HAS_FRAGMENT]->(F:Fragment)
            WITH count(f) as num_frag
            RETURN num_frag
            """, parameters={'data': self.dataset}).data()
        num_frag = int(pd.DataFrame(query_results)['num_frag'])
        return num_frag

    def prepareGraph(self):
        """
        Objective: Delete old weigths, Create difficulty score for fragments and HAS_FRAGMENT relationship
        :return:
        """

        if self.dataset != "Lipophilicity-ID.csv":
            query_results = self.graph.run("""
            // Make new weights for Dataset
                MATCH (D:DataSet{data: $data})-[:SPLITS_INTO_TEST]->(T:TestSet)-[p:CONTAINS_PREDICTED_MOLECULE]->(M:Molecule)-[f:HAS_FRAGMENT]->(F:Fragment)
                WITH avg(p.average_error) as difficulty, f, M, F
                SET M.difficulty = difficulty                      
                SET f.difficulty = difficulty
                RETURN M, F, f""", parameters={'data': self.dataset})
        else:
            query_results = self.graph.run(
                """
                // Make new weights for Dataset
                MATCH (D:DataSet{data: $data})-[:SPLITS_INTO_TEST]->(T:TestSet)-[p:CONTAINS_PREDICTED_MOLECULE]->(M:Molecule)-[f:HAS_FRAGMENT]->(F:Fragment)
                WHERE p.average_error < 10
                WITH avg(p.average_error) as difficulty, f, M, F
                SET M.difficulty = difficulty                      
                SET f.difficulty = difficulty
                RETURN M, F, f""", parameters={'data': self.dataset})

    def fragmentAnalysis(self, cutoff=0.9, easy_frag_lim=1000, hard_frag_lim=1000):
        """
        Objective: Data analysis on fragment difficulty
        :param cutoff:
        :param easy_frag_lim:
        :param hard_frag_lim:
        :return: a CSV file
        """
        csv_string = "FragAnalysis_" + self.dataset[:-4] + "_" + str(cutoff) + "_" \
                     + str(int(easy_frag_lim)) + "_" + str(int(hard_frag_lim)) + ".csv"

        query_results = self.graph.run(
            """
        // Remove Common Fragments
        MATCH (D:DataSet{data: $data})-[c:CONTAINS_MOLECULE]->(M:Molecule)
        WITH  percentileCont(M.difficulty, $cutoff) as cutoff

        MATCH (D:DataSet{data: $data})-[c:CONTAINS_MOLECULE]->(eM:Molecule)-[ef:HAS_FRAGMENT]->(eF:Fragment)
        WHERE eM.difficulty < cutoff // easy molecules
        WITH eF, count(ef) as efreq, cutoff // gath frags and frequency
        ORDER BY efreq DESC LIMIT $easy_frag_lim  //  limit to top n
        WITH  collect(eF) as easyFrags, cutoff

        MATCH (D:DataSet{data: $data})-[c:CONTAINS_MOLECULE]->(hM:Molecule)-[hf:HAS_FRAGMENT]->(hF:Fragment)
        WHERE hM.difficulty > cutoff // hard molecules
        WITH hF, count(hf) as hfreq, easyFrags
        ORDER BY hfreq DESC LIMIT $hard_frag_lim
        WITH collect(hF) as hardFrags, easyFrags

        // use APOC to do list intersect & subtraction
        WITH apoc.coll.intersection(easyFrags, hardFrags) as overlap, apoc.coll.subtract(hardFrags, easyFrags) as remain 

        // Find Molecule-Fragment pairs that have the remaining fragments and are in the dataset
        UNWIND remain as rFrags
        MATCH (D:DataSet{data: $data})-[c:CONTAINS_MOLECULE]->(M:Molecule)-[f:HAS_FRAGMENT]->(rFrags)
        WITH M, rFrags
        MATCH (M)-[f:HAS_FRAGMENT]->(rFrags)

        // Get Difficulty Stats for Remaining Fragments
        WITH rFrags.name as fragment, count(f) as number_of_rel, sum(f.difficulty) as sum_difficulty,sum(f.difficulty)/count(f) as avg_difficulty// , M, rFrags,f 
        RETURN fragment, number_of_rel, sum_difficulty, avg_difficulty
        ORDER BY number_of_rel DESC, avg_difficulty DESC""", parameters={'data': self.dataset, 'cutoff': cutoff,
                                                                         'easy_frag_lim': easy_frag_lim,
                                                                         'hard_frag_lim': hard_frag_lim }).data()
        df = pd.DataFrame(query_results)


        df.to_csv(csv_string)

    def cleanUp(self):
        """
        Clean up a second time
        :return:
        """
        self.graph.run(
            """// Delete old weights
                MATCH (M:Molecule)-[f:HAS_FRAGMENT]->(F:Fragment)
                REMOVE M.difficulty, f.difficulty                      
                RETURN M, F, f""")

if __name__ == '__main__':
    results_folder = 'results'
    check_for_results_folder(results_folder)

    dataset = ['water-energy.csv']
    cutoffs = [0.7, 0.9]
    easy_frag_limits = [0.1, 0.3]  # Percentage of fragment for easy fragment
    hard_frag_limits = [0.1, 0.4]  # Percentage of fragment for hard fragment
    with cd(ROOT_DIR + '/dataFiles/'):  # Initialize model
        for data in dataset:
            auto = CypherAutomate(dataset=data)
            auto.connect_to_neo4j(port="bolt://localhost:7687", username="neo4j", password="password")  # Connect to Neo4j
            auto.check_for_dataset()  # Check for current dataset in database
            num_frag = auto.countFrag()  # Count number of fragments for the current dataset
            auto.cleanUp()  # Clean up weights
            auto.prepareGraph()  # Prepare graphs with weights
            for cutoff in cutoffs:
                for i in easy_frag_limits:
                    for j in hard_frag_limits:
                        with cd(str(pathlib.Path(__file__).parent.absolute()) + "/" + results_folder):
                            auto.fragmentAnalysis(cutoff=cutoff, easy_frag_lim=int(num_frag*i), hard_frag_lim=int(num_frag*j))

            auto.cleanUp()  # Second clean up
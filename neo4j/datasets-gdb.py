"""
To create a graph database in Neo4j that contains our datasets.
Molecules should be nodes and datasets should be nodes.  They should be related.
Molecules should have properties that are RDKit features.
"""

from py2neo import Graph, Node, Relationship





graph = Graph("bolt://localhost:7687", user="neo4j", password="goku")


"""
Need to create formatted csv files for nodes and relationships for use in apoc.import.csv

Header for node would look something like this:
smiles:ID(Molecule-ID),name:STRING,MolWt:INT, logP:FLOAT

Header for relationship file would look like this
:START_ID(Molecule-ID),:END_ID(Molecule-ID),solubility:FLOAT

"""



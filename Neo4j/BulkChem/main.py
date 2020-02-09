from Neo4j.BulkChem.ChemNeo4j_v3 import create_relationships
from Neo4j.BulkChem.backends import init_neo_bulkchem

init_neo_bulkchem()
create_relationships('Default', retrieve_data_from_neo4j=True)
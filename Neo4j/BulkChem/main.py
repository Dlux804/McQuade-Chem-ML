from Neo4j.BulkChem.ChemNeo4j_v2 import init_neo_bulkchem, create_relationships

init_neo_bulkchem()
create_relationships('B', retrieve_data_from_neo4j=True)
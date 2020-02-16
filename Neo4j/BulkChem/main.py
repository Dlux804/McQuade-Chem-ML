from Neo4j.BulkChem.ChemNeo4j_v4 import create_relationships
from Neo4j.BulkChem.backends import init_neo_bulkchem

init_neo_bulkchem(fragments_as_nodes=False)
create_relationships('pka')
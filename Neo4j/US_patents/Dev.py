from Draw import drawer
from Neo4j.US_patents.backends import map_rxn_functional_groups

rxn_smarts = ("OC1O[C@H](O)[C@H](O)[C@H]1CO.C1=CC=CC=C1CCl>"
              "OS(=O)[O-].[Na+].BrBr.C(=O)(O)[O-].[Na+].[H]O[H].CCO>"
              "O=C1[C@H](OCC2=CC=CC=C2)[C@H](OCC3=CC=CC=C3)[C@@H](COCC4=CC=CC=C4)O1")

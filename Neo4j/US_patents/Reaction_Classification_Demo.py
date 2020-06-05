import pandas as pd
from Neo4j.US_patents.backends import map_rxn_functional_groups, classify_reaction
from Draw.drawer import save_reaction_image, save_rdkit_reaction_image

reaction_smiles_dict = dict(

    reaction_A_1=('O=C1[C@H](OCC2=CC=CC=C2)[C@H](OCC3=CC=CC=C3)[C@@H](COCC4=CC=CC=C4)O1.NC1=NC=NN2C1=CC=C2I>'
                  'C[Si](C)(Cl)C.CC([Mg]Cl)C.Cl[Mg]C1=CC=CC=C1.C1CCCO1>'
                  'OC1(C2=CC=C3N2N=CN=C3N)[C@H](OCC4=CC=CC=C4)[C@H](OCC5=CC=CC=C5)[C@@H](COCC6=CC=CC=C6)O1'),

    reaction_A_2=('OC1(C2=CC=C3N2N=CN=C3N)[C@H](OCC4=CC=CC=C4)[C@H](OCC5=CC=CC=C5)[C@@H](COCC6=CC=CC=C6)O1>'
                  'C[Si](C)(C#N)C.O=S(C(F)(F)F)(O)=O.C[Si](C)(OS(=O)(C(F)(F)F)=O)C.ClCCl>'
                  'NC1=NC=NN2C([C@@]3(C#N)[C@H](OCC4=CC=CC=C4)[C@H](OCC5=CC=CC=C5)[C@@H](COCC6=CC=CC=C6)O3)=CC=C21'),

    reaction_A_3=('NC1=NC=NN2C([C@@]3(C#N)[C@H](OCC4=CC=CC=C4)[C@H](OCC5=CC=CC=C5)[C@@H](COCC6=CC=CC=C6)O3)=CC=C21>'
                  'ClB(Cl)Cl.ClCCl>'
                  'O[C@H]1[C@@H](O)[C@@](C2=CC=C3N2N=CN=C3N)(C#N)O[C@@H]1CO'),

    reaction_A_4=('O[C@H]1[C@@H](O)[C@@](C2=CC=C3N2N=CN=C3N)(C#N)O[C@@H]1CO>'
                  'CC(OC)(OC)C.O=S(O)(O)=O.CC(C)=O>'
                  'OC[C@H]1O[C@](C2=CC=C3N2N=CN=C3N)(C#N)[C@H]4[C@@H]1OC(C)(C)O4'),

    reaction_A_5=('OC[C@H]1O[C@](C2=CC=C3N2N=CN=C3N)(C#N)[C@H]4[C@@H]1OC(C)(C)O4.O=[P@](N[C@@H](C)C(OCC(CC)CC)=O)'
                  '(OC1=CC=CC=C1)OC2=CC=C([N+]([O-])=O)C=C2>'
                  'Cl[Mg]Cl.CCN(C(C)C)C(C)C.CC#N>'
                  'NC1=NC=NN2C([C@@]3(C#N)[C@H](O)[C@H](O)[C@@H](CO[P@@](N[C@@H](C)C(OCC(CC)CC)=O)'
                  '(OC4=CC=CC=C4)=O)O3)=CC=C21'),

    reaction_B_1=("C(Cl)(Cl)(Cl)Cl.O=C(OC)OC>"
                  "ClCl>"
                  "O=C(OC(Cl)(Cl)Cl)OC(Cl)(Cl)Cl"),

    reaction_B_2=("O=C(OC(Cl)(Cl)Cl)OC(Cl)(Cl)Cl.CC(C)(O)C>"
                  "CCN(CC)CC.CCCCCC.[Na]O>"
                  "O=C(OC(OC(C)(C)C)=O)OC(C)(C)C"),

    reaction_B_3=("O=C(OC(OC(C)(C)C)=O)OC(C)(C)C.NN>"
                  "CC(O)C.ClCCl>"
                  "O=C(NN)OC(C)(C)C"),

    reaction_C_1=("C1=CC=CO1.OC>"
                  "[Cl][Cl].C[N+](CC)(CC)CC.[Cl-].[Na+].[Na+].[O-]C([O-])=O>"
                  "COC1C=CC(OC)O1"),

    reaction_C_2=("COC1C=CC(OC)O1>"
                  "[Na+].[Na+].[O-]C([O-])=O.[H][H].[Ni].[Al].OC>"
                  "COC1CCC(OC)O1"),

    reaction_D_1=("COC1OC(OC)CC1.CC(OC(NN)=O)(C)C>"
                  "[H][Cl].C1COCCO1>"
                  "CC(OC(NN1C=CC=C1)=O)(C)C"),

    reaction_D_2=("CC(OC(NN1C=CC=C1)=O)(C)C.CC#N>"
                  "O=C=NS(=O)(Cl)=O.CN(C)C=O>"
                  "CC(OC(NN1C(C#N)=CC=C1)=O)(C)C"),

    reaction_D_3=("CC(OC(NN1C(C#N)=CC=C1)=O)(C)C>"
                  "C1COCCO1.[H][Cl]>"
                  "NN1C(C#N)=CC=C1"),

    reaction_D_4=("NN1C(C#N)=CC=C1.CC(=O)O.C(=N)N>"
                  "[O-]P(=O)([O-])[O-].[K+].[K+].[K+].CCO>"
                  "NC1=NC=NN2C1=CC=C2"),

    reaction_D_5=("NC1=NC=NN2C1=CC=C2>"
                  "CCO.O=C(C(C)(C)N1Br)N(Br)C1=O>"
                  "NC1=NC=NN2C1=CC=C2Br"),

    reaction_E=("OC1O[C@H](O)[C@H](O)[C@H]1CO.C1=CC=CC=C1CCl>"
                "OS(=O)[O-].[Na+].BrBr.C(=O)(O)[O-].[Na+].[H]O[H].CCO>"
                "O=C1[C@H](OCC2=CC=CC=C2)[C@H](OCC3=CC=CC=C3)[C@@H](COCC4=CC=CC=C4)O1"),

)

fragments_df = pd.read_csv('datafiles/rxn_map_groups.csv')

for reaction_name, reaction_smiles in reaction_smiles_dict.items():
    frags = map_rxn_functional_groups(reaction_smiles, fragments_df=fragments_df)
    classification = classify_reaction(reaction_smiles, fragments_df=fragments_df)
    print(f'{reaction_name}: {classification}, [{frags}]')

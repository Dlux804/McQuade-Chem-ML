import pandas as pd
from rdkit import Chem
from io import StringIO


class Search_Fragments:

    @staticmethod
    def __gen_fragments_dict__():

        raw_frag_csv = StringIO("""Fragment,SMARTS,Sub-Fragment,Sub-SMARTS,
Aldehyde,[CX3H1](=O)[#6],,,
Ester,[#6][CX3](=O)[OX2H0][#6],Ether,[OD2]([#6])[#6],
Amide,[NX3][CX3](=[OX1])[#6],,,
Carbamate,"[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",,,
Amine,"[NX3;H2,H1;!$(NC=O)]",,,
Imine,"[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",,,
Arene,c,,,
Thiol,[#16X2H],,,
Acyl Halide,"[CX3](=[OX1])[F,Cl,Br,I]",Alkyl Halide,"[#6][F,Cl,Br,I]",
Allenic Carbon,[$([CX2](=C)=C)],,,
Vinylic Carbon,[$([CX3]=[CX3])],,,
Ketone,[#6][CX3](=O)[#6],,,
Carboxylic Acid,[CX3](=O)[OX2H1],Alcohol,[OX2H],
Thioamide,[NX3][CX3]=[SX1],,,
Nitrate,"[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",,,
Nitrile,[NX1]#[CX2],,,
Phenol,[OX2H][cX3]:[c],,,
Peroxide,"[OX2,OX1-][OX2,OX1-]",,,
Sulfide,[#16X2H0],,,
Nitro,"[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",,,
Phosphoric Acid,"[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",Phosphoric Ester,"[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        """)

        smarts = pd.read_csv(raw_frag_csv)
        smarts = smarts.to_dict('record')
        rdkit_smarts = []
        for i in range(len(smarts)):  # For all the functional groups
            working_pair = smarts[i]
            del working_pair['Unnamed: 4']  # Odd bug, delete random column that is produced
            working_pair['mol'] = Chem.MolFromSmarts(
                working_pair['SMARTS'])  # Make a new column, and add in the data Rdkit can interpet
            if str(working_pair['Sub-SMARTS']) != 'nan':
                working_pair['sub-mol'] = Chem.MolFromSmarts(
                    working_pair['Sub-SMARTS'])  # This part will search for Sub-SMARTS
            rdkit_smarts.append(working_pair)
        return rdkit_smarts

    '''
    @staticmethod
    def __gen_fragments_dict__():
        smarts = pd.read_excel("Function-Groups-SMARTS.xlsx")
        smarts = smarts.to_dict('record')
        rdkit_smarts = []
        for i in range(len(smarts)):  # For all the functional groups
            working_pair = smarts[i]
            working_pair['mol'] = Chem.MolFromSmarts(
                working_pair['SMARTS'])  # Make a new column, and add in the data Rdkit can interpet
            if str(working_pair['Sub-SMARTS']) != 'nan':
                working_pair['sub-mol'] = Chem.MolFromSmarts(
                    working_pair['Sub-SMARTS'])  # This part will search for Sub-SMARTS
            rdkit_smarts.append(working_pair)

        print(rdkit_smarts)
        return rdkit_smarts
    '''

    def search_fragments(self):
        results = []
        for i in range(len(self.all_fragments)):
            frag = self.all_fragments[i]
            if str(frag['Sub-SMARTS']) != 'nan':
                sub = len(self.mol.GetSubstructMatches(frag['sub-mol']))
                main = len(self.mol.GetSubstructMatches(frag['mol']))
                if sub != 0 and main != 0:
                    if sub != main:
                        results.append(frag['Fragment'])
                        results.append(frag['Sub-Fragment'])
                    else:
                        results.append(frag['Fragment'])
                elif sub != 0:
                    results.append(frag['Sub-Fragment'])
            else:
                main = len(self.mol.GetSubstructMatches(frag['mol']))
                if main != 0:
                    results.append(frag['Fragment'])
        return results

    def results_to_string(self):
        results = self.results
        results_string = ''
        for i in range(len(results)):
            if i == 0:
                results_string = results[i]
            else:
                results_string = results_string + ", " + results[i]
        return results_string

    def __init__(self, smiles):
        self.smiles = Chem.MolFromSmiles(smiles)
        self.con_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        self.mol = Chem.MolFromSmiles(smiles)
        self.all_fragments = self.__gen_fragments_dict__()
        self.results = self.search_fragments()
        self.results_string = self.results_to_string()
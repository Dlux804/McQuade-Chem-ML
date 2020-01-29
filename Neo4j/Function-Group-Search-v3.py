import pandas as pd
from rdkit import Chem


class Search_Fragments:

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
        return rdkit_smarts

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
        self.smiles = smiles
        self.con_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        self.mol = Chem.MolFromSmiles(smiles)
        self.all_fragments = self.__gen_fragments_dict__()
        self.results = self.search_fragments()
        self.results_string = self.results_to_string()


print(Search_Fragments('CCO[P](=S)(OCC)Oc1ccc(cc1)[N+]([O-])=O').results)
print(Search_Fragments('CCO[P](=S)(OCC)Oc1ccc(cc1)[N+]([O-])=O').results_string)

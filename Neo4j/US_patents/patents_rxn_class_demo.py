import pandas as pd
from Neo4j.US_patents.backends import classify_reaction
from Draw.drawer import save_reaction_image
from rdkit import Chem

reaction_smiles_dict = dict(

    reaction_1=('[Br:19][C&H2:20][C&H1:21](-,:[O&H1:32])[C&H2:22][C&H2:23][C&H2:24][C&H2:25][C&H2:26][C&H2:27]'
                '[C&H2:28][C&H2:29][C&H2:30][C&H3:31]'
                '>O.O.[#24](-,:O[#24](-,:[O&-])(=O)=O)(-,:[O&-])(=O)=O.[#11&+].[#11&+].CCOCC.CC(-,:C)=O.O.S(=O)(=O)'
                '(-,:O)O>'
                '[Br:19][C&H2:20][C:21](=[O:32])[C&H2:22][C&H2:23][C&H2:24][C&H2:25][C&H2:26][C&H2:27][C&H2:28]'
                '[C&H2:29][C&H2:30][C&H3:31]'),

    reaction_2=('[C&H1:1]1(-,:[N&H1:4][C:5](=[O:31])[C:6]2[C&H1:11]=[C&H1:10][C:9](-,:[C:12]3[N:16]4[N:17]=[C:18]'
                '(-,:[C&H1:28]=[O:29])[C&H1:19]=[C:20](-,:[N&H1:21][C&H2:22][C&H2:23][C:24](-,:[F:27])(-,:[F:26])'
                '[F:25])[C:15]4=[N:14][C&H1:13]=3)=[C&H1:8][C:7]=2[C&H3:30])[C&H2:3][C&H2:2]1.Br[#12][C:34]1'
                '[C&H1:39]=[C&H1:38][C:37](-,:[O:40][C&H3:41])=[C:36](-,:[F:42])[C&H1:35]=1'
                '>O1CCCC1.O.BrC1C=CC(-,:OC)=C(-,:F)C=1.[#12].[Cl&-].[N&H4&+]>'
                '[C&H1:1]1(-,:[N&H1:4][C:5](=[O:31])[C:6]2[C&H1:11]=[C&H1:10][C:9](-,:[C:12]3[N:16]4[N:17]=[C:18]'
                '(-,:[C&H1:28](-,:[C:34]5[C&H1:39]=[C&H1:38][C:37](-,:[O:40][C&H3:41])=[C:36](-,:[F:42])[C&H1:35]=5)'
                '[O&H1:29])[C&H1:19]=[C:20](-,:[N&H1:21][C&H2:22][C&H2:23][C:24](-,:[F:25])(-,:[F:26])[F:27])[C:15]'
                '4=[N:14][C&H1:13]=3)=[C&H1:8][C:7]=2[C&H3:30])[C&H2:2][C&H2:3]1'),

    reaction_3=('[Br:1][C:2]1[C&H1:3]=[C:4](-,:Br)[C:5]2[N:6](-,:[C:8](-,:[I:11])=[C&H1:9][N:10]=2)[N:7]=1.[F:13]'
                '[C:14](-,:[F:19])(-,:[F:18])[C&H2:15][C&H2:16][N&H2:17]>'
                'CN(-,:C)C=O.O>'
                '[Br:1][C:2]1[C&H1:3]=[C:4](-,:[N&H1:17][C&H2:16][C&H2:15][C:14](-,:[F:19])(-,:[F:18])[F:13])[C:5]2'
                '[N:6](-,:[C:8](-,:[I:11])=[C&H1:9][N:10]=2)[N:7]=1'),

    reaction_4=('[C&-]#N.[C:1](-,:[O:5][C:6](-,:[N:8]1[C&H2:13][C&H2:12][C&H1:11](-,:[S:14](-,:[C:16]2[C&H1:21]='
                '[C&H1:20][C:19](-,:Br)=[C&H1:18][C&H1:17]=2)=[O:15])[C&H2:10][C&H2:9]1)=[O:7])(-,:[C&H3:4])'
                '(-,:[C&H3:3])[C&H3:2]'
                '>[C&H3:23][N:24](-,:C=O)C.C1(-,:P(-,:C2C=CC=CC=2)[C&-]2C=CC=C2)C=CC=CC=1.[C&-]1(-,:P'
                '(-,:C2C=CC=CC=2)C2C=CC=CC=2)C=CC=C1.[#26&+2].C1C=CC(/C=C/C(/C=C/C2C=CC=CC=2)=O)=CC=1.'
                'C1C=CC(/C=C/C(/C=C/C2C=CC=CC=2)=O)=CC=1.C1C=CC(/C=C/C(/C=C/C2C=CC=CC=2)=O)=CC=1.[#46].[#46].'
                '[#30&+2].[C&-]#N'
                '>[C:1](-,:[O:5][C:6](-,:[N:8]1[C&H2:13][C&H2:12][C&H1:11](-,:[S:14](-,:[C:16]2[C&H1:21]=[C&H1:20]'
                '[C:19](-,:[C:23]#[N:24])=[C&H1:18][C&H1:17]=2)=[O:15])[C&H2:10][C&H2:9]1)=[O:7])(-,:[C&H3:4])'
                '(-,:[C&H3:3])[C&H3:2]'),

    reaction_5=('[O:1]1[C:5]2=[C&H1:6][N:7]=[C&H1:8][C&H1:9]=[C:4]2[C&H1:3]=[C:2]1[C:10](-,:[N&H1:12][C&H2:13][C:14]1'
                '[C&H1:19]=[C&H1:18][C:17](-,:[S:20](-,:Cl)(=[O:22])=[O:21])=[C&H1:16][C&H1:15]=1)=[O:11].[N:25]1'
                '(-,:[C&H1:30]2[C&H2:35][C&H2:34][N&H1:33][C&H2:32][C&H2:31]2)[C&H2:29][C&H2:28][C&H2:27][C&H2:26]1'
                '>C(-,:Cl)Cl.Cl.C(-,:N(-,:CC)CC)C'
                '>[N:25]1(-,:[C&H1:30]2[C&H2:35][C&H2:34][N:33](-,:[S:20](-,:[C:17]3[C&H1:18]=[C&H1:19][C:14]'
                '(-,:[C&H2:13][N&H1:12][C:10](-,:[C:2]4[O:1][C:5]5=[C&H1:6][N:7]=[C&H1:8][C&H1:9]=[C:4]5'
                '[C&H1:3]=4)=[O:11])=[C&H1:15][C&H1:16]=3)(=[O:22])=[O:21])[C&H2:32][C&H2:31]2)[C&H2:29][C&H2:28]'
                '[C&H2:27][C&H2:26]1')

)

fragments_df = pd.read_csv('datafiles/rxn_map_groups.csv')

for reaction_name, reaction_smiles in reaction_smiles_dict.items():
    classification = classify_reaction(reaction_smiles, fragments_df=fragments_df)
    print(f'{reaction_name}: {classification}')

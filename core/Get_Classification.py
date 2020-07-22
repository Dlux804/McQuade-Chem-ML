def get_classification_targets(data):
    if data == 'sider.csv':
        targets = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
                   'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
                   'Gastrointestinal disorders',
                   'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
                   'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
                   'General disorders and administration site conditions', 'Endocrine disorders',
                   'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                   'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders',
                   'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders',
                   'Psychiatric disorders', 'Renal and urinary disorders',
                   'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders',
                   'Cardiac disorders',
                   'Nervous system disorders', 'Injury, poisoning and procedural complications']
    elif data == 'clintox.csv':
        targets = ['FDA_APPROVED', 'CT_TOX']
    elif data == 'BBBP.csv':
        targets = 'p_np'
    elif data == 'bace.csv':
        targets = 'Class'
    else:
        return Exception(f"Dataset {data} unknown")

    return targets


def get_classification_feats(alg):
    if alg == 'svc':  # Normalized
        feats = [[1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1], [2], [3], [4],
                 [5], [6]]
    if alg == 'knc':  # Normalized
        feats = [[1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1], [2], [3], [4],
                 [5], [6]]
    if alg == 'rf':  # Not Normalized
        feats = [[0], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0], [2], [3], [4],
                 [5], [6]]

    return feats
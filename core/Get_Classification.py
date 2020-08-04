def get_classification_targets(data):
    if data == 'sider.csv':
        targets = ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Product issues',
                   'Eye disorders', 'Investigations', 'Musculoskeletal and connective tissue disorders',
                   'Gastrointestinal disorders',
                   'Social circumstances', 'Immune system disorders', 'Reproductive system and breast disorders',
                   'Neoplasms benign, malignant and unspecified',
                   'General disorders and administration site conditions', 'Endocrine disorders',
                   'Surgical and medical procedures', 'Vascular disorders', 'Blood and lymphatic system disorders',
                   'Skin and subcutaneous tissue disorders', 'Congenital, familial and genetic disorders',
                   'Infections and infestations', 'Respiratory, thoracic and mediastinal disorders',
                   'Psychiatric disorders', 'Renal and urinary disorders',
                   'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders',
                   'Cardiac disorders',
                   'Nervous system disorders', 'Injury, poisoning and procedural complications']
    if data == 'clintox.csv':
        targets = ['FDA_APPROVED', 'CT_TOX']
    if data == 'BBBP.csv':
        targets = 'p_np'
    if data == 'bace.csv':
        targets = 'Class'

    return targets




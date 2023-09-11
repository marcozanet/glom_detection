import numpy as np
import random
import pandas as pd


def _generate_random_patients_feats(n_patients:int):
    d_feat_range = {'age_at_biopsy':list(range(40, 100)), 'vasculitis_type':['mpo', 'pr3'], 'sex':['m', 'f'], 
                    'ethnicity':['caucasian', 'asian'], 'on_medications':[True, False], 'small_anca_type':['gpa', 'mpa', 'egpa'],
                    'perc_scler_gloms':list(np.arange(0,0.2,0.01)), 'perc_cellcresc_gloms':list(np.arange(0,2,0.01)), 
                    'perc_fibrcresc_gloms':list(np.arange(0,3,0.01)), 'perc_healthy_gloms':list(np.arange(0.4, 0.7, 0.01)), 
                    'berden_score':['Sclerotic', 'Focal', 'Mixed', 'Crescentic'], 'label':[1,2,3,4]}
    # d_feat_range = {'on_medications':[True, False], 'small_anca_type':['gpa', 'mpa', 'egpa'],
    #                 'perc_scler_gloms':list(np.arange(0,0.2,0.01)),  
    #                 'berden_score':['Sclerotic', 'Focal', 'Mixed', 'Crescentic'], 'label':[1,2,3,4]}
    patients = {}
    range_pat_ids = (104800, 209600)
    for i in range(n_patients):
        assert all([isinstance(val, list) for val in d_feat_range.values()]), f"All ranges in 'd_feat_range':{d_feat_range} should be type list."
        feats = {name:None for name in d_feat_range.keys()}
        for feat_n, values in d_feat_range.items():
            rand_val = random.choices(population=values)[0]
            if isinstance(rand_val, float): rand_val = round(rand_val, 2)
            feats[feat_n]= rand_val
        patients[random.randint(range_pat_ids[0], range_pat_ids[1])]=feats
    
    return patients


def _save_csv(patients:dict):
    df = pd.DataFrame(data=patients).transpose()
    cols_skip_onehot = [ 'age_at_biopsy', 'perc_scler_gloms','perc_cellcresc_gloms', 'perc_fibrcresc_gloms', 'perc_healthy_gloms', 'label']
    df_add_later = df.copy()[cols_skip_onehot]
    df = pd.get_dummies(df.drop(cols_skip_onehot, axis=1))
    df_final = df.join(df_add_later)
    print(df_final.columns)
    df_final.to_csv('patients_data.csv', index=True)

    return



if __name__=='__main__':
    n_patients = 300
    patients = _generate_random_patients_feats(n_patients=n_patients)
    _save_csv(patients=patients)

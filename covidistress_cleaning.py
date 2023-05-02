import pandas as pd
import numpy as np
import os 

# Change as needed
dir = '/Users/jamiezhang/Desktop/COVIDiSTRESS/'

new_cols = ['marital_status', 
    *[f"pss10_{i}" for i in range(1,11)], 
    *[f"lon_{i}" for i in range(1,4)], 
    *[f"sps_{i}" for i in range(1,11)]
    ]

old_cols = ['Dem_maritalstatus', 
            *[f"Scale_PSS10_UCLA_{i}" for i in range(1, 11)], 
            *[f"Scale_SLON_{i}" for i in range(1,4)], 
            *[f"SPS_{i}" for i in range(1,11)]
            ]

old_cols_april = ['Dem_maritalstatus', 
            *[f"Scale_PSS10_UCLA_{i}" for i in range(1, 11)], 
            *[f"Scale_Lon_{i}" for i in range(1,4)], 
            *[f"SPS_{i}" for i in range(1,11)]
            ]

reverse_cols = ['pss10_4_reversed', 
                'pss10_5_reversed',
                'pss10_7_reversed',
                'pss10_8_reversed'
                ]

def clean(dfs, months): 
    cleaned_dfs = []
    for i, df in enumerate(dfs): 
        if months[i] == 'April': 
            tmp_df = df.loc[:,old_cols_april].replace(0, np.nan)

        else: 
            tmp_df = df.loc[:,old_cols].replace(0, np.nan)
        tmp_df = tmp_df.dropna()
        tmp_df.columns = new_cols
        cleaned_dfs.append(tmp_df)
    return cleaned_dfs

def compute_scores(dfs): 
    scored_dfs = []
    for df in dfs: 
        scored_df = df
        # pss
        pss_df = df.iloc[:,2:12]
        not_reverse = pss_df.iloc[:,[0,1,2,5,8,9]] 
        reverse = np.abs((pss_df.iloc[:,[3,4,6,7]]-5)) 
        scored_df[reverse_cols] = reverse
        pss_reverse = pd.concat([not_reverse, reverse], axis=1)
        pss_composite = np.sum(pss_reverse, axis=1)#.reset_index(drop=True)
        scored_df['pss10_composite'] = pss_composite

        # loneliness
        lon_df = df.iloc[:,12:15]
        lon_composite = np.sum(lon_df, axis=1)
        scored_df['lon_composite'] = lon_composite

        # social support
        sps_df = df.iloc[:,15:25]
        sps_composite = np.sum(sps_df, axis=1)
        scored_df['sps_composite'] = sps_composite

        scored_dfs.append(scored_df) 
    return scored_dfs

def main():
    dfs = []
    names = []
    months = []

    # Create dataframes for each month
    for file in os.listdir(dir):
        if (not file.startswith('~') and 
            not file.startswith('clean') and
            file.endswith('.csv')): 
            dfs.append(pd.read_csv(file, index_col=0, encoding='ISO-8859-1'))
            names.append(file)
            months.append(file.split('.')[0])

    # Clean data & compute composite scores
    cleaned_dfs = clean(dfs, months)
    composite_dfs = compute_scores(cleaned_dfs)

    # Create csv files
    for i, df in enumerate(composite_dfs): 
        df.to_csv('clean_' + names[i])   
      
if __name__ == '__main__':
    main()
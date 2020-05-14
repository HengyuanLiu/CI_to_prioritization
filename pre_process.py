import os
import pandas as pd
import tc_data_reconstitution as tcDataRecon

path = './data'
dirs = os.listdir(path)

for dataset in dirs:
    dataset_name = dataset[0:-8]
    tc_data = pd.read_csv(path + '/' + dataset, sep=';', index_col='Id')
    tc_data = tcDataRecon.tc_data_generalize_Stage(tc_data)
    tc_data.to_csv(path + '/' + dataset_name + '.csv', sep=';')

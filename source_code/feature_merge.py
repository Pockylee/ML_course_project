import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('-t', '--fp_type', type=str, action='store', default='all', help='Fingerprint type that user defined.(maccs / ecfp / all/ checkmol)')
parser.add_argument('-m'    , '--merge_list'    , type=str, action='store', default='all', help='Generate all Fingerprint type. ( y / n )')
parser.add_argument('-i'    , '--input_dir_path'    , type=str, action='store', default='/data/poyili/UnderG/model_test/20211202/', help='Generate all Fingerprint type. ( y / n )')
args = parser.parse_args()

checkmol_file_name = 'checkmol.csv'
pubchem_file_name = 'pubchem.csv'
inhouse_file_name = 'inhouse.csv'
ring_file_name = 'ring.csv'
maccs_file_name = 'maccs.csv'
ecfp_file_name = 'ecfp.csv'

Merge_list = []
if args.merge_list == 'all':
	Merge_list = ['checkmol', 'pubchem', 'inhouse', 'ring', 'maccs', 'ecfp']

else:
	Merge_list = args.merge_list.strip().split(",")
	#for merge_target in args.merge_list.strip().split(","):
	#Merge_list.append(merge_target)

print(Merge_list)

df_checkmol = pd.DataFrame()
df_pubchem = pd.DataFrame()
df_inhouse = pd.DataFrame()
df_ring = pd.DataFrame()
df_maccs = pd.DataFrame()
df_ecfp = pd.DataFrame()


Dict = {'checkmol':df_checkmol,
'pubchem':df_pubchem,
'inhouse':df_inhouse,
'ring':df_ring,
'maccs':df_maccs,
'ecfp':df_ecfp
}

Smile_list = []
Concat_list = []
output_file_type_name = ''
for feature_type in Merge_list:
	file_name = feature_type + '.csv'
	Dict[feature_type] = pd.read_csv(os.path.join(args.input_dir_path, file_name), header=None)
	header_list = ['SMILES']
	for i in range(len(Dict[feature_type].columns)):
	    if i == 0:
	        continue
	    header_list.append(feature_type + '#' + str(i - 1))
	Dict[feature_type].columns = header_list
	Smile_list = Dict[feature_type]['SMILES']
	del Dict[feature_type]['SMILES']
	Concat_list.append(Dict[feature_type])
	output_file_type_name += feature_type[0]
	#print(Dict[feature_type])

#df_ecfp = pd.read_csv("/data/poyili/UnderG/model_test/20211202/ecfp.csv")

#print(Dict['ecfp'])


df_all = pd.concat(Concat_list, axis=1)
df_all.insert(0, "SMILES",Smile_list)
print(df_all)
if args.merge_list != 'all':
	output_file_name = 'merged_features_' + output_file_type_name + '.csv'
else:
	output_file_name = 'merged_features_all.csv'
df_all.to_csv(os.path.join(args.input_dir_path, output_file_name), sep=',', index=False, header=True)
#smile_list = df_ecfp['SMILES']
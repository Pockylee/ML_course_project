import glob
import rdkit
import os
import argparse
import subprocess

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#parser.add_argument('-t', '--fp_type', type=str, action='store', default='all', help='Fingerprint type that user defined.(maccs / ecfp / all/ checkmol)')

parser.add_argument('-checkmol', '--checkmol', type=str, action='store', default='n', help='Generate Checkmol Fingerprint type. ( y / n )')
parser.add_argument('-pubchem' , '--pubchem' , type=str, action='store', default='n', help='Generate PubChem Fingerprint type. ( y / n )')
parser.add_argument('-inhouse' , '--inhouse' , type=str, action='store', default='n', help='Generate PubChem Fingerprint type. ( y / n )')
parser.add_argument('-ring'    , '--ring'    , type=str, action='store', default='n', help='Generate PubChem Fingerprint type. ( y / n )')
parser.add_argument('-maccs'   , '--maccs'   , type=str, action='store', default='n', help='Generate MACCS Fingerprint type. ( y / n )')
parser.add_argument('-ecfp'    , '--ecfp'    , type=str, action='store', default='n', help='Generate ECFP Fingerprint type. ( y / n )')

parser.add_argument('-f'       , '--input_format'    , type=str, action='store', default='smiles', help='Format of input files. (smile / sdf)')
#parser.add_argument('-i'       , '--input_jobid'  , type=str, action='store', help='Job id of input directory.')
parser.add_argument('-p'       , '--input_path'  , type=str, action='store',default='/data/poyili/UnderG/model_test/20211202', help='input directory.')


#parser.add_argument('-o', '--output_dir_path', type=str, action='store', default='./test_output', help='Path of output directory.')
#parser.add_argument('-ce', '--output_csv_ecfp', type=str, action='store', default='all_ECFP4_fea.csv', help='File name of the output ECFP4 CSV file.')
#parser.add_argument('-cm', '--output_csv_maccs', type=str, action='store', default='all_MACCS_fea.csv', help='File name of the output MACCS CSV file.')
args = parser.parse_args()

#Check whether the output directory is already exist or not
#if not os.path.exists(args.output_dir_path):
#    os.makedirs(args.output_dir_path)

print("Input file format : " + args.input_format)
#print("Input Job ID : " + args.input_jobid)
print("Checkmol : " + args.checkmol)
print("PubChem : " + args.pubchem)
print("In-house : " + args.inhouse)
print("Ring in drug : " + args.ring)
print("MACCS : " + args.maccs)
print("ECFP : " + args.ecfp)

#input_dir_path = './data_tmp/' + args.input_jobid
input_dir_path = args.input_path


pubchem_moieties = glob.glob("./generate/pcfp_ring_mol/*.mol")
inhouse_moieties = glob.glob("./generate/inhouse_moiety_mol/*.mol")
ring_moieties = glob.glob("./generate/ring_drug_mol/*.mol")


#If the user input the argument -F with sdf which means there are several or one sdf file in the input directory.
if args.input_format == "sdf":
    
    all_files = glob.glob(os.path.join(input_dir_path, "compound_file/*.sdf"))
    
    output_lines_checkmol = []
    output_lines_pubchem = []
    output_lines_inhouse = []
    output_lines_ring = []
    output_lines_ecfp = []

    maccs_fea_list = []
    
    compound_name_list = []
    

    #Go through all files in the input directory
    for file_index, file_name in enumerate(all_files):
        
        compound_name = os.path.basename(file_name)[:-4]
        compound_name_list.append(compound_name)
        
        mol = Chem.MolFromMolFile(file_name)
        img = Chem.Draw.MolToImageFile(mol , os.path.join(input_dir_path, 'compound_img/' + compound_name + '.png'), size=(500, 500))

        #ECFP4 feature part
        if args.ecfp == 'y':
            #mol = Chem.MolFromMolFile(file_name)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            Bits = rdkit.DataStructs.cDataStructs.BitVectToText(fingerprint)
            bits_array = []
            for bit in Bits:
                bits_array.append(bit)

            output_lines_ecfp.append(compound_name+','+','.join(bits_array)+'\n')

        #MACCS feature part
        if args.maccs == 'y':
            #mol = Chem.MolFromMolFile(file_name)
            fps = MACCSkeys.GenMACCSKeys(mol)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fps,fp_arr)
                
            if(fp_arr.sum != 0):
                maccs_fea_list.append(fp_arr)

        #Checkmol feature part
        if args.checkmol == 'y':
            
            checkmol_tmp_array = np.zeros(204)

            result = subprocess.run(['./generate/checkmol', '-p', file_name], stdout=subprocess.PIPE)
            output_cm_lines = result.stdout.splitlines()

            for output_cm_line in output_cm_lines:
                
                num_exp = output_cm_line.decode().split(':')    # num_exp[1] >= 1 -> have this feature
                fea_exp = num_exp[0].split('#')                 # fea_exp[1] -> feature num
                fea_num = int(fea_exp[1])

                if(int(num_exp[1]) >= 1):
                    checkmol_tmp_array[fea_num] = 1
            
            output_lines_checkmol.append(compound_name + ',' + ','.join([str(bit) for bit in checkmol_tmp_array]) + '\n')
            
        #PubChem feature part
        if args.pubchem == 'y':
            
            pubchem_tmp_array = np.zeros(168, int)

            for index, moiety in enumerate(pubchem_moieties):

                result = subprocess.run(['./generate/matchmol', moiety, file_name], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    pubchem_tmp_array[index] = 1

            output_lines_pubchem.append(compound_name + ',' + ','.join([str(bit) for bit in pubchem_tmp_array]) + '\n')

        #Inhouse feature part
        if args.inhouse == 'y':
            
            inhouse_tmp_array = np.zeros(34, int)

            for index, moiety in enumerate(inhouse_moieties):

                result = subprocess.run(['./generate/matchmol', moiety, file_name], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    inhouse_tmp_array[index] = 1

            output_lines_inhouse.append(compound_name + ',' + ','.join([str(bit) for bit in inhouse_tmp_array]) + '\n')

        #Ring feature part
        if args.ring == 'y':
            
            ring_tmp_array = np.zeros(147, int)

            for index, moiety in enumerate(ring_moieties):

                result = subprocess.run(['./generate/matchmol', moiety, file_name], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    ring_tmp_array[index] = 1

            output_lines_ring.append(compound_name + ',' + ','.join([str(bit) for bit in ring_tmp_array]) + '\n')
    
    #ECFP4 output part
    if args.ecfp == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'ecfp.csv'), 'w')
        output_lines_file.writelines(output_lines_ecfp)
        output_lines_file.close()

    #MACCS output part
    if args.maccs == 'y':
        output_lines_maccs = []        
        for index, compound_fea in enumerate(maccs_fea_list):
            compound_fea_int = compound_fea.astype(int)
            output_lines_maccs.append(compound_name_list[index]+','+','.join(map(str, compound_fea_int))+'\n')

        output_lines_file = open(os.path.join(input_dir_path, 'maccs.csv'), 'w')
        output_lines_file.writelines(output_lines_maccs)
        output_lines_file.close()

    #Checkmol output part
    if args.checkmol == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'checkmol.csv'), 'w')
        output_lines_file.writelines(output_lines_checkmol)
        output_lines_file.close()

    #PubChem output part
    if args.pubchem == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'pubchem.csv'), 'w')
        output_lines_file.writelines(output_lines_pubchem)
        output_lines_file.close()

    #Inhouse output part
    if args.inhouse == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'inhouse.csv'), 'w')
        output_lines_file.writelines(output_lines_inhouse)
        output_lines_file.close()

    #Ring output part
    if args.ring == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'ring.csv'), 'w')
        output_lines_file.writelines(output_lines_ring)
        output_lines_file.close()
    


#If the user input the argument -F with 'smiles' which means there shall be a txt file contains lines of smiles code in the input file.
elif args.input_format == 'smiles':
    #file input and preproccessing
    file = open(os.path.join(input_dir_path, 'compound_list.txt'))
    input_smiles_list = file.read().strip().split("\n")

    output_lines_checkmol = []
    output_lines_pubchem = []
    output_lines_inhouse = []
    output_lines_ring = []

    compound_name_list = []
    fea_list = []

    count_num = 0
    #get every smiles code's morgan fingerprint
    for input_smile in input_smiles_list:
        count_num += 1
        print(count_num, end="\r")
        
        mol = Chem.MolFromSmiles(input_smile)
        molblock = Chem.MolToMolBlock(mol)
        input_smile_file = input_smile.replace("/", "_")
        mol_path = os.path.join(input_dir_path, 'compound_file/' + input_smile_file + '.mol')


        #img = Chem.Draw.MolToImageFile(mol , os.path.join(input_dir_path, 'compound_img/' + input_smile_file + '.png'), size=(500, 500))

        compound_name_list.append(input_smile)

        #with open(mol_path, "w") as smiles2mol:
        #    smiles2mol.write(molblock)

        if args.ecfp == 'y':
            #mol = Chem.MolFromSmiles(input_smile)
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            
            #convert into numpy array
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fingerprint, fp_arr)

            if input_smiles_list.index(input_smile) == 0:
                fp_matrix = fp_arr
            else:
                fp_matrix = np.vstack((fp_matrix, fp_arr))


        if args.maccs == 'y':
            fps = MACCSkeys.GenMACCSKeys(mol)
            fp_arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fps,fp_arr)
                
            if(fp_arr.sum != 0):
                fea_list.append(fp_arr)


        if args.checkmol == 'y':
            
            checkmol_tmp_array = np.zeros(204)

            result = subprocess.run(['./generate/checkmol', '-p', mol_path], stdout=subprocess.PIPE)
            output_cm_lines = result.stdout.splitlines()

            for output_cm_line in output_cm_lines:
                
                num_exp = output_cm_line.decode().split(':')    # num_exp[1] >= 1 -> have this feature
                fea_exp = num_exp[0].split('#')                 # fea_exp[1] -> feature num
                fea_num = int(fea_exp[1])

                if(int(num_exp[1]) >= 1):
                    checkmol_tmp_array[fea_num] = 1

            output_lines_checkmol.append(input_smile + ',' + ','.join([str(bit) for bit in checkmol_tmp_array]) + '\n')


        if args.pubchem == 'y':
            
            pubchem_tmp_array = np.zeros(168, int)

            for index, moiety in enumerate(pubchem_moieties):

                result = subprocess.run(['./generate/matchmol', moiety, mol_path], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    pubchem_tmp_array[index] = 1

            output_lines_pubchem.append(input_smile + ',' + ','.join([str(bit) for bit in pubchem_tmp_array]) + '\n')


        if args.inhouse == 'y':
            
            inhouse_tmp_array = np.zeros(34, int)

            for index, moiety in enumerate(inhouse_moieties):

                result = subprocess.run(['./generate/matchmol', moiety, mol_path], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    inhouse_tmp_array[index] = 1

            output_lines_inhouse.append(input_smile + ',' + ','.join([str(bit) for bit in inhouse_tmp_array]) + '\n')

        
        if args.ring == 'y':
            
            ring_tmp_array = np.zeros(147, int)

            for index, moiety in enumerate(ring_moieties):

                result = subprocess.run(['./generate/matchmol', moiety, mol_path], stdout=subprocess.PIPE)
                result = str(result.stdout)

                if(result.find('T') != -1):
                    ring_tmp_array[index] = 1

            output_lines_ring.append(input_smile + ',' + ','.join([str(bit) for bit in ring_tmp_array]) + '\n')


    if args.maccs == 'y':
        output_lines_maccs = []        
        for index, compound_fea in enumerate(fea_list):
            compound_fea_int = compound_fea.astype(int)
            output_lines_maccs.append(input_smiles_list[index]+','+','.join(map(str, compound_fea_int))+'\n')

        output_lines_file = open(os.path.join(input_dir_path, 'maccs.csv'), 'w')
        output_lines_file.writelines(output_lines_maccs)
        output_lines_file.close()

    if args.ecfp == 'y':
        #save all features and smiles code name into a dataframe
        #print(fp_matrix)
        if count_num == 1:
            fp_matrix = [fp_matrix]
        df_fp_all = pd.DataFrame(fp_matrix)
        #print(input_smiles_list)
        #print(df_fp_all)
        df_fp_all.insert(loc=0, column='SMILES', value=input_smiles_list)

        #write the dataframe into a output file.
        df_fp_all.to_csv(os.path.join(input_dir_path, 'ecfp.csv'), sep=',', index=False, header=False)

    if args.checkmol == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'checkmol.csv'), 'w')
        output_lines_file.writelines(output_lines_checkmol)
        output_lines_file.close()

    if args.pubchem == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'pubchem.csv'), 'w')
        output_lines_file.writelines(output_lines_pubchem)
        output_lines_file.close()

    if args.inhouse == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'inhouse.csv'), 'w')
        output_lines_file.writelines(output_lines_inhouse)
        output_lines_file.close()

    if args.ring == 'y':
        output_lines_file = open(os.path.join(input_dir_path, 'ring.csv'), 'w')
        output_lines_file.writelines(output_lines_ring)
        output_lines_file.close()
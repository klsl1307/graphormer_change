import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import torch
import datetime

'''
Generate 3d coordinates for all molecules from smiles.
'''

smiles = pd.read_csv('/home/wzh/Graphormer-change/dataset/ogbg_molpcba/mapping/mol.csv.gz', compression='gzip', header = None, skiprows=1)[128].astype(str)

MAX_ATOM_SIZE = 300
pos = torch.zeros([smiles.shape[0],MAX_ATOM_SIZE,3]) # xyz, max atom size set as 200

time = datetime.datetime.now()
postfix = time.strftime(f"%m-%d_%H-%M-%S")
print(postfix)

for i in range(smiles.shape[0]):
    mol = Chem.MolFromSmiles(smiles[i])

    ## Generate 3D coordinates
    mol = Chem.AddHs(mol)  # Add hydrogens for a more realistic 3D structure

    ## way 1
    allconformers = AllChem.EmbedMultipleConfs(
            mol, numConfs=50, randomSeed=24, clearConfs=True
        )
    sz = len(allconformers)
    for j in range(sz):
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=i) # lowest energy
        except:
            continue

    ## way 2
    # res = AllChem.EmbedMolecule(mol,useRandomCoords=True)
    # if res < 0:
    #     print("Bad conformation at string "+str(i+1))
    #     continue

    mol = Chem.RemoveHs(mol)

    ## get conformer
    try:
        conformer = mol.GetConformer(0)
    except:
        print("Can't get conformation at string "+str(i+1))
        continue

    for atom_idx in range(mol.GetNumAtoms()):
        atom_pos = conformer.GetAtomPosition(atom_idx)
        # Convert the Point3D object to a NumPy array
        atom_pos_np = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
        # Assuming pos is a PyTorch FloatTensor
        pos[i,atom_idx,:] = torch.FloatTensor(atom_pos_np)

torch.save(pos, './pos2.pth')

time = datetime.datetime.now()
postfix = time.strftime(f"%m-%d_%H-%M-%S")
print(postfix)

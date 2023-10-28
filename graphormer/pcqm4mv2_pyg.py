import os
import os.path as osp
import shutil
from mol import smiles2graph, mol2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from rdkit import Chem
import random

class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root = '/data/wzh/mol/', smiles2graph = smiles2graph, mol2graph = mol2graph, transform=None, pre_transform = None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.mol2graph = mol2graph ## TODO
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1
        
        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        print('Finish download.')
        # if decide_download(self.url):
        #     # path = download_url(self.url, self.original_root)
        #     # extract_zip(path, self.original_root)
        #     # os.unlink(path)
        #     print('Finish download.')
        # else:
        #     print('Stop download.')
        #     exit(-1)

    def process(self):
        ## TODO: this is greatly changed

        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles'] # from the whole dataset, not all we want
        homolumogap_list = data_df['homolumogap'] # from the whole dataset, not all we want

        ## TODO:
        sdf_path = '/data/wzh/mol/pcqm4m-v2-train.sdf' # contain smiles & conformer
        sdf = Chem.SDMolSupplier(sdf_path)
        # new_idx = []
        # sdf_idx = []
        # for idx, mol in enumerate(sdf):
        #     smile = Chem.MolToSmiles(mol)
        #     df_idx = data_df[data_df['smiles'] == smile].index # idx of the matched smiles
        #     if not df_idx.empty: # may have empty match
        #         new_idx.append(df_idx[0])
        #         sdf_idx.append(idx)
            # if not df_idx.empty:
            #     homogap = data_df.loc[df_idx, 'homolumogap']

        print('Converting SMILES strings into graphs...')
        data_list = []

        # new_smile_list = smiles_list[new_idx]
        # new_homo_list = homolumogap_list[new_idx]
        # new_sdf = sdf[sdf_idx] # avoid empty matching

        for i in tqdm(range(3378606)): # length of the training set
            data = Data()

            smiles = smiles_list[i] # the form of smile is not the same as sdf! thus may generate inconsistent atom features

            homolumogap = homolumogap_list[i]
            # if sdf[i] is None:
            #     continue
            graph = self.mol2graph(sdf[i], smiles) # compute features based on mol (not smiles)
            
            assert(len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert(len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            
            data.smiles = smiles # TODO
            pos = sdf[i].GetConformer().GetPositions() # only pos from sdf
            data.pos = torch.from_numpy(pos).to(torch.float32)

            data_list.append(data)

        # torch.save(data_list,osp.join(self.folder, 'data_list.pt')) # save processed data
        # double-check prediction target
        # self.data_list = data_list
        split_dict = self.get_idx_split()
        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

    def get_idx_split(self):
        # split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt')))
        # split_dict = torch.load(osp.join(self.folder, 'new_split_dict.pt')) # change split
        # divide following 8:1:1
        split_dict = {}
        all_indices = list(range(self.data.y.shape[0]))

        # Shuffle the indices randomly
        random.shuffle(all_indices)

        # Calculate the sizes of the three splits based on the desired ratios
        total_samples = len(all_indices)
        train_ratio = 0.8  # 80% for training
        val_ratio = 0.1    # 10% for validation
        test_ratio = 0.1   # 10% for testing

        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        test_size = int(total_samples * test_ratio)

        # Divide the shuffled indices into the three splits
        split_dict['train'] = all_indices[:train_size]
        split_dict['valid'] = all_indices[train_size:train_size + val_size]
        split_dict['test'] = all_indices[train_size + val_size:]

        return split_dict

if __name__ == '__main__':
    dataset = PygPCQM4Mv2Dataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
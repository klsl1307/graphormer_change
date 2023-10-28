# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data import get_dataset
from lr import PolynomialDecayLR
import torch
import math
import torch.nn as nn
import pytorch_lightning as pl

from utils.flag import flag_bounded

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdDistGeom as molDG
import networkx as nx

import torch.nn.functional as F
import numpy as np

def init_bert_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

## TODO: get mol and position
def set_conformation(x, attn_edge_type):
    # x: Batch * atom * F
    # print(x[:,:,0])
    pos = torch.zeros([x.shape[0],x.shape[1],3]) # xyz
    for i in range(x.shape[0]):
        mol = Chem.RWMol()

        # reconstruct atoms of the mol in rdkit form, dont add H atom
        # atom_order_list = []
        for j in range(x.shape[1]):
            idx = x[i,j,0].item()
            if idx == 0: # not an atom
                break
            atom = Chem.Atom(idx-1) # first element is the atomic num - 1!!!
            atom.SetFormalCharge(x[i,j,3].item()-1543) # add charge!!!
            # atom_order_list.append(j)
            mol.AddAtom(atom)

        # reconstruct bonds of the mol in rdkit form
        # print(attn_edge_type.shape)
        adj = attn_edge_type[i,:,:,0] - 2 # 0-3 as the reasonable rdkit bond idx, bias 2-5
        edge_attr = torch.where(adj>-2)
        # print(edge_attr)

        type_list = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
        for j in range(edge_attr[0].shape[0]):
            start = edge_attr[0]
            end = edge_attr[1]
            if start[j]>=end[j]:
                continue      
            mol.AddBond(start[j].item(),end[j].item(),type_list[adj[start[j].item(),end[j].item()]])

        # add Hs
        try:
            Chem.SanitizeMol(mol)
        except ValueError as e:
            print(f"Error while sanitizing molecule: {str(e)}")
        Chem.Kekulize(mol, clearAromaticFlags=True, clearConjugatedFlags=True)
        mol = Chem.AddHs(mol)  # Don't Add hydrogens for a more realistic 3D structure
        AllChem.EmbedMolecule(mol, randomSeed=21)  # Generate 3D coordinates
        mol = Chem.RemoveHs(mol) # remove for pos
        conformer = mol.GetConformer(0)

        # save pos of every atom in mol
        # for j in range(x.shape[1]):
        #     if x[i,j,0].item() == 0: # not an atom
        #         continue
        #     pos[i,j] = mol.GetConformer(0).GetAtomPosition(j)

        for atom_idx in range(mol.GetNumAtoms()):
            atom_pos = conformer.GetAtomPosition(atom_idx)
            # Convert the Point3D object to a NumPy array
            atom_pos_np = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
            # Assuming pos is a PyTorch FloatTensor
            pos[i,atom_idx,:] = torch.FloatTensor(atom_pos_np)

    return pos

def set_conformation2(x, smiles):
    pos = torch.zeros([x.shape[0],x.shape[1],3]) # xyz

    for i in range(x.shape[0]):
        mol = Chem.MolFromSmiles(smiles[i])

        ## Generate 3D coordinates
        mol = Chem.AddHs(mol)  # Add hydrogens for a more realistic 3D structure
        # AllChem.EmbedMolecule(mol, randomSeed=1)  # Generate 3D coordinates
        AllChem.EmbedMolecule(mol, maxAttempts=5000)
        AllChem.UFFOptimizeMolecule(mol) # may help the bad conformation
        mol = Chem.RemoveHs(mol) # remove for pos

        ## from uni-mol, num_confs change from 1000 to 100
        # mol = Chem.AddHs(mol)
        # allconformers = AllChem.EmbedMultipleConfs(
        #     mol, numConfs=50, randomSeed=42, clearConfs=True
        # )
        # sz = len(allconformers)
        # for j in range(sz):
        #     try:
        #         AllChem.MMFFOptimizeMolecule(mol, confId=i) # lowest energy
        #     except:
        #         continue
        # mol = Chem.RemoveHs(mol)

        # Access the 3D coordinates
        conformer = mol.GetConformer(0)
        for atom_idx in range(mol.GetNumAtoms()):
            atom_pos = conformer.GetAtomPosition(atom_idx)
            # Convert the Point3D object to a NumPy array
            atom_pos_np = np.array([atom_pos.x, atom_pos.y, atom_pos.z])
            # Assuming pos is a PyTorch FloatTensor
            pos[i,atom_idx,:] = torch.FloatTensor(atom_pos_np)
            # pos[i,atom_idx,:] = atom_pos

    return pos

class NonLinear(nn.Module):
    def __init__(self, input, output_size, hidden=None):
        super(NonLinear, self).__init__()

        if hidden is None:
            hidden = input
        self.layer1 = nn.Linear(input, hidden)
        self.layer2 = nn.Linear(hidden, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x

def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)
    try:
        # will random generate conformer with seed equal to -1. else fixed random seed.
        res = AllChem.EmbedMolecule(mol, randomSeed=seed)
        if res == 0:
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        ## for fast test... ignore this ###
        elif res == -1 and mode == 'heavy':
            AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
            try:
                # some conformer can not use MMFF optimize
                AllChem.MMFFOptimizeMolecule(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            except:
                AllChem.Compute2DCoords(mol)
                coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
                coordinates = coordinates_2d
        else:
            AllChem.Compute2DCoords(mol)
            coordinates_2d = mol.GetConformer().GetPositions().astype(np.float32)
            coordinates = coordinates_2d
    except:
        print("Failed to generate conformer, replace with zeros.")
        coordinates = np.zeros((len(atoms),3))
    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h
    else:
        return atoms, coordinates

class Graphormer_0904(pl.LightningModule):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        flag=False,
        flag_m=3,
        flag_step_size=1e-3,
        flag_mag=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.alpha = 0.1
        self.beta = torch.nn.Parameter(torch.Tensor([1.]), requires_grad=True)

        self.num_heads = num_heads
        if dataset_name == 'ZINC':
            self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(
                    40 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(40, num_heads, padding_idx=0)
            self.in_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                64, hidden_dim, padding_idx=0)
        else:
            self.atom_encoder = nn.Embedding(
                512 * 9 + 1, hidden_dim, padding_idx=0)
            self.edge_encoder = nn.Embedding(
                512 * 3 + 1, num_heads, padding_idx=0)
            self.edge_type = edge_type
            if self.edge_type == 'multi_hop':
                self.edge_dis_encoder = nn.Embedding(
                    128 * num_heads * num_heads, 1)
            self.rel_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0) # TODO: change to float input
            self.rel_pos_encoder2 = nn.Linear(1,num_heads) # TODO: expand, may not be a good choice
            self.in_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)
            self.out_degree_encoder = nn.Embedding(
                512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        if dataset_name in ['PCQM4M-LSC','PCQM4Mv2-LSC']:
            self.out_proj = nn.Linear(hidden_dim, 1)
        else:
            self.downstream_out_proj = nn.Linear(
                hidden_dim, get_dataset(dataset_name)['num_class'])

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.evaluator = get_dataset(dataset_name)['evaluator']
        self.metric = get_dataset(dataset_name)['metric']
        self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.flag = flag
        self.flag_m = flag_m
        self.flag_step_size = flag_step_size
        self.flag_mag = flag_mag
        self.hidden_dim = hidden_dim
        self.automatic_optimization = not self.flag
        self.apply(lambda module: init_bert_params(module, n_layers=n_layers))

        ## TODO
        self.proj_layer = NonLinear(1, num_heads) # expand the last dimension

    def forward(self, batched_data, perturb=None):
        # rel_pos: SPD
        # x: Batch * Atom * F

        sample_idx = batched_data.idx # TODO: idx of samples

        attn_bias, rel_pos, x = batched_data.attn_bias, batched_data.rel_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        
        ## TODO: get_conformation: Batch * atom * 3
        # import pandas as pd
        # smiles_data = pd.read_csv('/home/wzh/Graphormer-change/dataset/ogbg_molpcba/mapping/mol.csv.gz', compression='gzip', header = None, skiprows=1)[128].astype(str)
        # smiles = smiles_data[sample_idx.tolist()]
        smiles = batched_data.smiles
        # batch_position = set_conformation2(x, smiles)
        # batch_position = set_conformation(x, attn_edge_type)

        ## TODO: new rel_pos by conformation
        # batch_num = batch_position.shape[0]
        # atom_num = batch_position.shape[1]
        # new_rel_pos = torch.zeros([batch_num,atom_num,atom_num])
        # for i in range(batch_num):
        #     for j in range(atom_num):
        #         for k in range(j+1,atom_num):
        #             distance = torch.norm(batch_position[i,j] - batch_position[i,k],p=2)
        #             new_rel_pos[i,j,k] = distance
        #             new_rel_pos[i,k,j] = distance
        # rel_pos = new_rel_pos ## TODO: novel change #1

        ## TODO: get adj matrix
        # rel_pos = torch.zeros([x.shape[0],x.shape[1],x.shape[1]])
        # for i in range(x.shape[0]):
        #     mol = Chem.MolFromSmiles(smiles[i])

        #     # try:
        #     #     AllChem.EmbedMolecule(mol)
        #     # except:
        #     #     print("Can't embed molecule at order "+str(i))

        #     mol = Chem.AddHs(mol)

        #     # Get order of non Hs
        #     H_order = []
        #     count = 0
        #     for atom in mol.GetAtoms():
        #         if atom.GetAtomicNum() != 1: 
        #             H_order.append(count)
        #         count = count + 1

        #     dm=AllChem.Get3DDistanceMatrix(mol)
        #     r_dm = dm[H_order,H_order] # remove Hs
        #     mol = Chem.RemoveHs(mol)
        #     r_dm = torch.from_numpy(r_dm)
        #     zeroPad = nn.ZeroPad2d(padding=(0, x.shape[1]-r_dm.shape[1], 0, x.shape[1]-r_dm.shape[1]))
        #     # zeroPad = np.pad(dm, ((0, x.shape[1]-dm.shape[1]),(0, x.shape[1]-dm.shape[1])), 'constant', constant_values=(0,0)) # padding
        #     rel_pos[i] = zeroPad

        ## TODO: get coordinates
        pos = batched_data.pos
        # pos = torch.zeros((x.shape[0],x.shape[1],3))
        # for i in range(x.shape[0]): 
        #     atom, coord = inner_smi2coords(smiles[i])
        #     coord = torch.from_numpy(coord)
        #     zeroPad = nn.ZeroPad2d(padding=(0, 0, 0, x.shape[1]-coord.shape[0]))
        #     pos[i] = zeroPad(coord)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        rel_pos = delta_pos.norm(dim=-1).view(-1, pos.shape[1], pos.shape[1])
        delta_pos /= rel_pos.unsqueeze(-1) + 1e-5 # normalize

        # rf_pred
        if self.dataset_name == 'ogbg-molhiv':
            mgf_maccs_pred = batched_data.y[:, 2]
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        ## TODO rel pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        device = torch.cuda.current_device()
        rel_pos = rel_pos.to(device) # [n_graph, n_node, n_node]

        rel_pos_bias = self.proj_layer(rel_pos.unsqueeze(-1)) # expand 1 dimension
        # rel_pos_bias = rel_pos_bias.view(n_graph, n_node, n_node, self.num_heads)

        rel_pos_bias = rel_pos_bias.permute(0, 3, 1, 2).contiguous()
        padding_mask = x.eq(0).all(dim=-1)
        rel_pos_bias.masked_fill_(
            padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
        )

        # rel_pos_bias = self.rel_pos_encoder2(rel_pos.unsqueeze(-1)).permute(0, 3, 1, 2)

        ## TODO change
        # tmp = torch.normal(0,0.1,rel_pos_bias.shape)
        # device = torch.cuda.current_device()
        # tmp = tmp.to(device)
        # rel_pos_bias = rel_pos_bias + tmp

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                        :, 1:, 1:] + rel_pos_bias  # spatial encoder
        # reset rel pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            rel_pos_ = rel_pos.clone()
            rel_pos_[rel_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            rel_pos_ = torch.where(rel_pos_ > 1, rel_pos_ - 1, rel_pos_)
            if self.multi_hop_max_dist > 0:
                rel_pos_ = rel_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(
                3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) /
                          (rel_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(
                attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                        :, 1:, 1:] + edge_input  # edge encoder
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        node_feature = self.atom_encoder(x).sum(
            dim=-2)           # [n_graph, n_node, n_hidden]
        if self.flag and perturb is not None:
            node_feature += perturb

        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)  # add degree encoder
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        # output part
        if self.dataset_name in ['PCQM4M-LSC','PCQM4Mv2-LSC']:
            # get whole graph rep
            output = self.out_proj(output[:, 0, :])
        else:
            output = self.downstream_out_proj(output[:, 0, :])  # virtual node
            # graph_pred =  torch.sigmoid(output)
            # mgf_maccs_pred = torch.sigmoid(mgf_maccs_pred)

            # h_graph_final = torch.cat((graph_pred, mgf_maccs_pred.reshape(-1,1)), 1)
            # att = torch.nn.functional.softmax(h_graph_final * self.beta, -1)
            # output = torch.sum(h_graph_final * att, -1).reshape(-1,1)

            if self.dataset_name == 'ogbg-molhiv':
                output = torch.sigmoid(output)
                mgf_maccs_pred = torch.sigmoid(mgf_maccs_pred)
                output = torch.clamp((1 - self.alpha) * output + self.alpha * mgf_maccs_pred.reshape(-1,1), min=0, max=1)
        
        return output

    def training_step(self, batched_data, batch_idx):
        if self.dataset_name == 'ogbg-molpcba':
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                # y_gt = batched_data.y.view(-1).float()
                y_gt = batched_data.y[:, :128].reshape(-1, 1).view(-1).float()
                mask = ~torch.isnan(y_gt)
                loss = self.loss_fn(y_hat[mask], y_gt[mask])
            else:
                # y_gt = batched_data.y.view(-1).float()
                y_gt = batched_data.y[:, :128].reshape(-1, 1).view(-1).float()
                mask = ~torch.isnan(y_gt)

                def forward(perturb): return self(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt[mask], optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag, mask=mask)
                self.lr_schedulers().step()

        elif self.dataset_name == 'ogbg-molhiv':  # batched_data.y[:,0]
            if not self.flag:
                y_hat = self(batched_data).view(-1)
                # y_gt = batched_data.y.view(-1).float()
                y_gt = batched_data.y[:, 0].float()
                loss = self.loss_fn(y_hat, y_gt)
            else:
                # y_gt = batched_data.y.view(-1).float()
                y_gt = batched_data.y[:, 0].float()
                def forward(perturb): return self(batched_data, perturb)
                model_forward = (self, forward)
                n_graph, n_node = batched_data.x.size()[:2]
                perturb_shape = (n_graph, n_node, self.hidden_dim)

                optimizer = self.optimizers()
                optimizer.zero_grad()
                loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
                                       m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
                self.lr_schedulers().step()
        else:
            y_hat = self(batched_data).view(-1)
            y_gt = batched_data.y.view(-1)
            # y_gt = batched_data.y[:, 0]
            loss = self.loss_fn(y_hat, y_gt)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC','PCQM4Mv2-LSC','ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
            # y_true = batched_data.y[:, 0]
        else:
            if self.dataset_name == 'ogbg-molhiv':
                y_pred = self(batched_data)
                y_true = batched_data.y[:, 0:1]
            else:
                y_pred = self(batched_data)
                y_true = batched_data.y[:, 0:128]
            # y_pred = self(batched_data).view(-1)
            # y_true = batched_data.y[:, 0]
        return {
            'y_pred': y_pred,
            'y_true': y_true,
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        # if self.dataset_name == 'ogbg-molpcba':
        #     mask = ~torch.isnan(y_true)
        #     loss = self.loss_fn(y_pred[mask], y_true[mask])
        #     self.log('valid_ap', loss, sync_dist=True)
        # else:
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        try:
            self.log('valid_' + self.metric, self.evaluator.eval(input_dict)
                        [self.metric], sync_dist=True)
        except:
            pass

    def test_step(self, batched_data, batch_idx):
        if self.dataset_name in ['PCQM4M-LSC','PCQM4Mv2-LSC','ZINC']:
            y_pred = self(batched_data).view(-1)
            y_true = batched_data.y.view(-1)
        else:
            if self.dataset_name == 'ogbg-molhiv':
                y_pred = self(batched_data)
                y_true = batched_data.y[:, 0:1]
            else:
                y_pred = self(batched_data)
                y_true = batched_data.y[:, 0:128]
            # y_pred = self(batched_data).view(-1)
            # y_true = batched_data.y[:, 0]
        return {
            'y_pred': y_pred,
            'y_true': y_true,
            'idx': batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        if self.dataset_name in ['PCQM4M-LSC','PCQM4Mv2-LSC']:
            result = y_pred.cpu().float().numpy()
            idx = torch.cat([i['idx'] for i in outputs])
            torch.save(result, 'y_pred.pt')
            torch.save(idx, 'idx.pt')
            exit(0)
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        self.log('test_' + self.metric, self.evaluator.eval(input_dict)
                 [self.metric], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--num_heads', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate',
                            type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        parser.add_argument('--flag', action='store_true')
        parser.add_argument('--flag_m', type=int, default=3)
        parser.add_argument('--flag_step_size', type=float, default=1e-3)
        parser.add_argument('--flag_mag', type=float, default=1e-3)
        parser.add_argument('--use_fps', type=bool, default=True)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

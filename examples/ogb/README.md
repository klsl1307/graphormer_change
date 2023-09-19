# Open Graph Benchmark

[https://arxiv.org/abs/2005.00687] (https://arxiv.org/abs/2005.00687)

[https://ogb.stanford.edu/](https://ogb.stanford.edu/)

## Results

#### OGBG-MolPCBA
Method        | #params | test AP (%)|
--------------|---------|------------|
DeeperGCN-VN+FLAG         | 5.6M    | 28.42      |
DGN          | 6.7M    | 28.85      |
GINE-VN          | 6.1M    | 29.17      |
PHC-GNN          | 1.7M    | 29.47      |
GINE-APPNP          | 6.1M    | 29.79      |
Graphormer   | 119.5M  | **31.39**      |

#### OGBG-MolHIV
Method        | #params | test ROC-AUC (%)|
--------------|---------|------------|
GCN-GraphNorm          | 526K    | 78.83      |
PNA          | 326K    | 79.05      |
PHC-GNN          | 111K    | 79.34      |
DeeperGCN-FLAG          | 532K    | 79.42      |
DGN          | 114K    | 79.70      |
Graphormer   | 47.0M   | **80.51**      |
Graphormer + FPs   | 47.0M   | **82.25**      |



| Model                | Test ROC-AUC   | Valid ROC-AUC  | Parameters | Hardware          |
| -------------------- | --------------- | --------------- | ---------- | ----------------- |
| Graphormer + FPs | 82.25 ± 0.01 | 83.96 ± 0.01 | 47085378     | Tesla V100 (32GB) |


## Example Usage

Prepare your pre-trained models following our paper ["Do Transformers Really Perform Bad for Graph Representation?"](https://arxiv.org/abs/2106.05234).

```bash
# Pre-train model for OGBG-Molhiv
bash ../ogb-lsc/lsc-hiv.sh
# Pre-train model for OGBG-Molpcba
bash ../ogb-lsc/lsc-pcba.sh
```
**The pre-trained model should be saved in `../../checkpoints/xxx.ckpt` manually.**

Fine-tuning your pre-trained model on OGBG-MolPCBA:

```bash
bash pcba.sh
```

Fine-tuning your pre-trained model on OGBG-MolHIV with **FingerPrints**:

First, you should generate fingerprints and train with random forest as mentioned in [《Extended-Connectivity Fingerprints》](https://pubs.acs.org/doi/10.1021/ci100050t) and [《GMAN and bag of tricks for graph classification》](https://github.com/PierreHao/YouGraph/blob/main/report/GMAN%20and%20bag%20of%20tricks%20for%20graph%20classification.pdf).

```bash
python extract_fingerprint.py
python random_forest.py
```
The random forest results will be saved in `../../rf_preds_hiv/rf_final_pred.npy`.

Then, you can fine-tune our pre-trained model. we use fingerprints to smooth the final results like APPNP.

```bash
bash hiv.sh
```
You can change `seed` in `hiv.sh`, and run 10 times.

Some hyper-parameters:

```bash
Namespace(accelerator='ddp', accumulate_grad_batches=1, amp_backend='native', amp_level='O2', attention_dropout_rate=0.1, auto_lr_find=False, auto_scale_batch_size=False, auto_select_gpus=False, batch_size=128, benchmark=False, check_val_every_n_epoch=1, checkpoint_callback=True, checkpoint_path='../../checkpoints/PCQM4M-LSC-epoch=192-valid_mae=0.1298.ckpt', dataset_name='ogbg-molhiv', default_root_dir='../../exps/hiv/hiv_flag/4', deterministic=False, distributed_backend=None, dropout_rate=0.1, edge_type='multi_hop', end_lr=1e-09, fast_dev_run=False, ffn_dim=768, flag=True, flag_m=2, flag_mag=0.0, flag_step_size=0.2, flush_logs_every_n_steps=100, gpus=2, gradient_clip_algorithm='norm', gradient_clip_val=0.0, hidden_dim=768, intput_dropout_rate=0.0, limit_predict_batches=1.0, limit_test_batches=1.0, limit_train_batches=1.0, limit_val_batches=1.0, log_every_n_steps=50, log_gpu_memory=None, logger=True, max_epochs=6, max_steps=645, max_time=None, min_epochs=None, min_steps=None, move_metrics_to_cpu=False, multi_hop_max_dist=5, multiple_trainloader_mode='max_size_cycle', n_layers=12, num_heads=32, num_nodes=1, num_processes=1, num_sanity_val_steps=2, num_workers=8, overfit_batches=0.0, peak_lr=0.0002, plugins=None, precision=16, prepare_data_per_node=True, process_position=0, profiler=None, progress_bar_refresh_rate=10, rel_pos_max=1024, reload_dataloaders_every_epoch=False, replace_sampler_ddp=True, resume_from_checkpoint=None, seed=4, stochastic_weight_avg=False, sync_batchnorm=False, terminate_on_nan=False, test=False, tot_updates=644, tpu_cores=None, track_grad_norm=-1, truncated_bptt_steps=None, val_check_interval=1.0, validate=False, warmup_updates=64, weight_decay=0.0, weights_save_path=None, weights_summary='top')
```

## Citation
Please kindly cite this paper if you use the code:
```
@article{ying2021transformers,
  title={Do Transformers Really Perform Bad for Graph Representation?},
  author={Ying, Chengxuan and Cai, Tianle and Luo, Shengjie and Zheng, Shuxin and Ke, Guolin and He, Di and Shen, Yanming and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2106.05234},
  year={2021}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

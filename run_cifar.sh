#!/bin/bash
#resnet32 resnet44 resnet56 resnet110 resnet1202

looper=1;
bn=0;
lr=$lr;
bt=$bt;
subset=$subset;
epochs=$epochs;
sample_mode=$sample_mode;
algo=$algo;
wd=$wd;
arch="vgg_manual";
loss=$loss;
init_scale=1;  
train_final_layer=0;

act='gelu';
cfg='AV';



python -u Cifar10_runs.py  \
--arch=$arch  \
--save-dir=$save_dir \
--weight-decay=$wd \
--data_subset=$subset \
--loss_type=$loss \
--augment=0 \
--epochs=$epochs \
--lr=$lr  \
--optimizer=$algo \
--batch-size=$bt  \
--looper=$looper \
--final_layer_init_scale=$init_scale \
--train_final_layer=$train_final_layer \
--sample_mode=$sample_mode \
--act=$act \
--bn=$bn \
--vgg_cfg=$cfg  \
--wandb_project=$wandb_project \
--compute_top_eigenval=1;


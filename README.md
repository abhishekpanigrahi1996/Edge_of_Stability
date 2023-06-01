# Understanding Gradient Descent on Edge of Stability in Deep Learning
This is the codebase of the paper [Understanding Gradient Descent on Edge of Stability
in Deep Learning](https://arxiv.org/pdf/2205.09745.pdf).



## Requirements

We provide a yml file that can be used to create a conda environment, containing all the necessary packages.

## Installation
Install necessary conda environment using 
conda env create -n EoSenv --file environment.yml



## Reproducing Figures 1 and 2
Figure1.ipynb and Figure2.ipynb contain the necessary code to reproduce the figures 1 and 2 in the main paper.


## Experiments with VGG on CIFAR10
To run normalized GD and GD with sqrt loss, please look at submit_EoS.sh and run_EoS.sh. The overview of the command in run_EoS.sh is as follows.

```bash
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
```

* `save_dir`: Path to save the checkpoints
* `wd`: weight decay (set as 0 in the paper)
* `subset`: Number of training examples (set as 5000 in the paper)
* `loss_type`: 'mse' for normalizedgd, 'sqrtmse' for GD with sqrt loss
* `epochs`: Number of training epochs 
* `lr`: Learning Rate (0.25 in the paper)
* `algo`: sgd for both normalizedgd and GD with sqrt loss
* `bt`: Batch size of dataloader
* `init_scale`: Initialization scale of final layer of VGG (1 for the paper)
* `train_final_layer`: Whether to train final layer of VGG (0 for the paper)
* `sample_mode`: Sampling mode for dataloader (random_shuffling for the paper)
* `act`: activation function (gelu for our paper)
* `bn`: 0 since we never use batch norm
* `cfg`: Configuration of VGG (AV for the paper)
* `wandb_project`: Weights and bias project to store the results to


## Experiments with on normalized GD and log sharpness minimizer on MNIST
To run normalized GD and GD with sqrt loss, please look at submit_mnist.sh and run_mnist.sh. The overview of the command in run_mnist.sh is as follows.
```bash
python mnist_runs.py \
--lr $lr \
--activation gelu \
--algo $algo \
--max_training_examples 1000 \
--batch-size 1000 \
--criterion $criterion \
--save-model \
--epochs $epochs  \
--save_freq $save_freq \
--resume $resume_checkpoint \
--hess_freq=$hess_freq \
--log_sharpness=$log_sharpness \
--start 1 \
--wandb_project $wandb_project\
--save_dir $save_dir\
--projection_lr 0.001; 
```

* `save_dir`: Path to save the checkpoints
* `criterion`: 'mse' for all GD, normalizedgd, regularizer_minimizer
* `epochs`: Number of training epochs  (10000 for GD, 120000 for normalized gradient descent, 300 for regularizer_minimizer)
* `lr`: Learning Rate (0.01 for GD, normalizedgd, regularizer_minimizer in our paper)
* `algo`: GD for gradient descent, normalizedgd for normalized gradient descent, regularizer_minimizer for sharpness minimizer
* `save_freq`: Save frequency of checkpoints (1000 for GD, and normalized gradient descent, 10 for regularizer_minimizer)
* `hess_freq`: Frequency of hessian computations 
* `log_sharpness`: 1 for normalized gd, since the regularizer should minimize log of hessian sharpness
* `wandb_project`: Weights and bias project to store the results to
* `projection_lr`: Learning rate for projection to manifold per regularizer minimizer step
* `start`: Resume from a previous run, if necessary 
* `resume`: resume from a checkpoint (for normalized gd and regularizer_minimizer, we need to start from the best checkpoint of gradient descent)


## Bugs and questions?
If you have any questions related to the code, feel free to email Abhishek (`{ap34}@cs.princeton.edu`). If you encounter a problem or bug when using the code, you can also open an issue.


## Citation

Please cite our work if you make use of our code or our pre-computed kernels in your work:

```bibtex
@inproceedings{arora2022understanding,
  title={Understanding gradient descent on the edge of stability in deep learning},
  author={Arora, Sanjeev and Li, Zhiyuan and Panigrahi, Abhishek},
  booktitle={International Conference on Machine Learning},
  pages={948--1024},
  year={2022},
  organization={PMLR}
}
```


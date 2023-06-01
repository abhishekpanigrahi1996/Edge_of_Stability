




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
--wandb_project=$wandb_project \
--save_dir=$save_dir \
--projection_lr=0.001;

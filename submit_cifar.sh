for k in 0; do
    lr=.25\
    subset=5000\
    wd=0\
    sample_mode='random_shuffling'\
    bt=5000\
    epochs=20000\
    algo='normalizedgd'\
    loss='mse'\
    wandb_project='Edge_of_Stability'\
    save_dir='Checkpoints/cifar_normalizedgd'\
    bash run_cifar.sh;
done



for k in 0; do
    lr=.25\
    subset=5000\
    wd=0\
    sample_mode='random_shuffling'\
    bt=5000\
    epochs=20000\
    algo='sgd'\
    loss='sqrtmse'\
    wandb_project='Edge_of_Stability'\
    save_dir='Checkpoints/cifar_gdsqrtmse'\
    bash run_cifar.sh;
done
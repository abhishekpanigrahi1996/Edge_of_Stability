


for k in 0; do
    lr=0.01\
    algo='normalizedgd'\
    criterion='mse'\
    epochs=120000\
    save_freq=1000\
    hess_freq=4000\
    log_sharpness=1\
    wandb_project='Edge_of_Stability'\
    resume_checkpoint='Checkpoints/MNIST_GD_checkpoint-9001.th'\
    save_dir='Checkpoints/mnist_normalizedgd'\
    bash run_mnist.sh;
done



for k in 0; do
    lr=0.01\
    algo='regularizer_minimize'\
    criterion='mse'\
    epochs=300\
    save_freq=10\
    hess_freq=10\
    resume_checkpoint='Checkpoints/MNIST_GD_checkpoint-9001.th'\
    log_sharpness=1\
    wandb_project='Edge_of_Stability'\
    save_dir='Checkpoints/mnist_sharpnessminimizer'\
    bash run_mnist.sh;
done



# Notes on hyperparameters

## PPOConfig
batch_size: default 128
mini_batch_size: whatever fits?
gradient_accumilation_steps: ? shouldnt this be determined by minibatch size?
learning_rate: default 1.41e-5
adap_kl_control: ?
init_kl_coeff: default 0.2 (used for adaptive kl control)
kl_penalty: default kl (kl, abs, mse, full) ?
target: 6 (target kl value for adaptive kl control)
cliprange: default 0.2
cliprange_value: 0.2
vf_coeff: default 0.1
ppo_epochs: default 4
target_kl: default 1 (stops early if we exceed this by 50%)
compare_steps: default 1
ratio threshold: default 10 (skip minibatches with high PPO ratios that can cause high loss spikes)
use_score_scaling: default False

we support score (aka reward) scaling/normalization/clipping to improve training stability
score scaling
score normalizing
score clipping

optimize_cuda_cache: probably best to set to True

are these relevant for us?
gamma
lambda


## PPOTrainer
num_shared_layers: default none
lr_scheduler: default none (linear or cosine should be fine)
optimizer: default Adam (try AdamW)

## Generation Kwargs
top_k
top_p: 0.9
do_sample
temperature 0.7
repetition_penalty

## PEFT
r
lora_alpha
lora_dropout
bias

## Other
8bit or 4bit model? Is this referring to the reference model?
Adam8bit optimizer?

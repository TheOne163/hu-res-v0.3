#!/bin/sh
env="StarCraft2"
map="5m_vs_6m"
algo="rmappo"
exp="hu-res-5m6m"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python ../train/train_smac.py \
    --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --map_name ${map} --seed ${seed} \
    --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 \
    --num_env_steps 10000000 \
    --use_value_active_masks --use_eval --eval_episodes 32 \
    --episode_length 400 \
    --clip_param 0.05 \
    --ppo_epoch 10 \
    --enable_hu \
    --sub_episode_length 50 \
    --sub_ppo_epoch 4 \
    --sub_num_mini_batch 1 \
    --sub_lr_scale 1.0 \
    --sub_entropy_coef 0.015 \
    --enable_res_hu \
    --res_bias_scale 0.1 \
    --sub_clip_param 0.2
done
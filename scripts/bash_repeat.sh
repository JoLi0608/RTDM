#!/bin/bash


# Plot planet
#for d in /app/data/planet/default/* ;do
#    for n in $d/*; do
#        for s in $n/*/; do
#            echo $s
#            python /app/RTDM/scripts/benchmarkgpu.py --path $s --algo planet --cpu $1 --gpu $2
#            break
#        done
#    done
#done


# ARS

for d in /app/data/ray_results/*
do
    for env in continuous_CartPole-v0 Hopper-v2 Humanoid-v2 HalfCheetah-v2 Pusher-v2
    do
        for n in $d/ARS_$env*/ ; do
          echo $n
          python /app/RTDM/scripts/evaluate_repeat.py --path $n --algo ars --evaseed 1 &
        done
        wait
    done
done



# SAC
for d in /app/data/spinup/sac/* ;do
    for n in $d/cmd_sac_pytorch/*/; do
        echo $n
        python /app/RTDM/scripts/evaluate_repeat.py --path $n --algo sac --evaseed 1 &
    done
    wait
done

# MBPO
for d in /app/data/mbpo/default/*/ ;do
    for n in $d*/; do
        for s in $n*/; do
            echo $s
            python /app/RTDM/scripts/evaluate_repeat.py --path $n --algo mbpo --evaseed 1 &
        done
        wait
    done
done


# PPO
for d in /app/data/spinup/ppo/*
do
    for n in $d/cmd_ppo_pytorch/*/
    do
        echo $n
        python /app/RTDM/scripts/evaluate_repeat.py --path $n --algo ppo --evaseed 1 &
    done
    wait
done





# PETS
for d in /app/data/pets/*/ ;do
    for n in $d*/; do
        echo $n
        python /app/RTDM/scripts/evaluate_repeat.py --path $n --algo pets --evaseed 1 &
    done
    wait
done

# RTRL


for d in /app/data/rtrl_3/exp/1-*/ ;do
    echo $d
    python /app/RTDM/scripts/evaluate_repeat.py --path $n --algo rtrl --evaseed 1
done
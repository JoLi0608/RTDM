
#!/bin/bash

# Plot planet
for d in /app/data/planet/default/* ;do
    for n in $d/*; do
        for s in $n/*/; do
            echo $s
            python /app/RTDM/scripts/plot_results.py --path $s --algo planet
        done
    done
done

# Dreamer
for d in /app/data/dreamer/logdir/* ;do
    for n in $d/dreamerv2/*/; do
        echo $n
        python /app/RTDM/scripts/plot_results.py --path $n --algo dreamer
    done
done

# SAC
for d in /app/data/spinup/sac/*/ ;do
    for n in $d/cmd_sac_pytorch/*/; do
        python /app/RTDM/scripts/plot_results.py --path $n --algo sac
    done
done

# MBPO
for d in /app/data/mbpo/default/*/ ;do
    for n in $d*/; do
        for s in $n*/; do
            echo $s
            python /app/RTDM/scripts/plot_results.py --path $s --algo mbpo
        done
    done
done

# PETS
for d in /app/data/pets/*/ ;do
    for n in $d*/; do
        echo $n
        python /app/RTDM/scripts/plot_results.py --path $n --algo pets
    done
done

# PPO
for d in /app/data/spinup/ppo/*
do
    for n in $d/cmd_ppo_pytorch/*/
    do
        echo $n
        python /app/RTDM/scripts/plot_results.py --path $n --algo ppo
    done
done

# ARS

for d in /app/data/ray_results/*
do
    for env in continuous_CartPole-v0 Hopper-v2 Humanoid-v2 HalfCheetah-v2 Pusher-v2
    do
        for n in $d/ARS_$env*/ ; do
          echo $n
          python /app/RTDM/scripts/plot_results.py --path $n --algo ars

        done
    done
done





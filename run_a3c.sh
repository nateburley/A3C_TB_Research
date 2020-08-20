#!/bin/bash
currentDate=`date +"%m-%d-%Y"`

# Two games to choose from: MsPacman, Pong
game=$1

if [ "$game" == "MsPacman" ]; then
    echo "Training Ms.Pac-Man..."
elif [ "$game" == "Pong" ]; then
    echo "Training Pong..."
else
    echo "Invalid game! Exiting..."; exit
fi

python3 DeepRL/run_experiment.py \
  --gym-env=${game}NoFrameskip-v4 \
  --parallel-size=16 \
  --max-time-step-fraction=0.5 \
  --use-mnih-2015 --input-shape=88 --padding=SAME \
  --checkpoint-freq=5 \
  --append-experiment-num=${currentDate} \

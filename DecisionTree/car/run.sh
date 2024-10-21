#!/bin/bash
gcc car_learner.c -lm -o car_learner.out
gcc car_tree.c -o car_tree.out
printf "\n-- Running tree learning algorithm on car training data --\n\n"
./car_learner.out
printf '\n-- Running predictor algorithm --\n\n'
./car_tree.out
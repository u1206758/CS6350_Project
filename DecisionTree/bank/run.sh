#!/bin/bash
gcc bank_learner.c -lm -o bank_learner.out
gcc bank_tree.c -o bank_tree.out
printf "\n-- Running tree learning algorithm on bank training data --\n\n"
./bank_learner.out
printf '\n-- Running predictor algorithm --\n\n'
./bank_tree.out
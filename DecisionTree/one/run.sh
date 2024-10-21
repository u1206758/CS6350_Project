#!/bin/bash
gcc learner.c -lm -o learner.out
gcc tree.c -o tree.out
printf "\n-- Running tree learning algorithm on salary training data --\n\n"
./learner.out
printf '\n-- Running predictor algorithm --\n\n'
./tree.out
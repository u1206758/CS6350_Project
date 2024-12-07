#!/bin/bash
gcc standard_perceptron.c -lm -o standard_perceptron.out
gcc voted_perceptron.c -lm -o voted_perceptron.out
gcc average_perceptron.c -lm -o average_perceptron.out
printf "\n-- Running standard perceptron --\n\n"
./standard_perceptron.out
printf "\n-- Running voted perceptron --\n\n"
./voted_perceptron.out
printf "\n-- Running average perceptron --\n\n"
./average_perceptron.out
rm *.out

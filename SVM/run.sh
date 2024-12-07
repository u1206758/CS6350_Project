#!/bin/bash
gcc SVM.c -lm -o SVM.out
printf "\n-- Running SVM --\n\n"
./SVM.out
rm *.out

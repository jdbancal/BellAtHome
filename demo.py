#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 00:09:11 2022

@author: bancal
"""

from bah import BellAtHome

### Parameters ###


# Number of rounds
n = 10000

# Maximum bias allower per individual raw random bit
epsilon = 0.2

# Number of raw random bits used to create one question
k = 1


### Script ###


# Run Alice
alice = BellAtHome(n = n, epsilon = epsilon, k = k, device='Alice')
alice.generateRandomness()
alice.computeQuestions()
alice.answerQuestions()

# Run Bob
bob = BellAtHome(n = n, epsilon = epsilon, k = k, device='Bob')
bob.generateRandomness()
bob.computeQuestions()
bob.answerQuestions()

# Check actual bias
print(f'Average bias of Alice = {alice.computeAverageBias()}')
print(f'Average bias of Bob = {bob.computeAverageBias()}')

# Compute CHSH score
print(f'Observed CHSH score = {bob.computeCHSH()}')


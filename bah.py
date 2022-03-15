#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:20:42 2022

@author: bancal
"""

import os
import numpy as np

from utils_io import writeDataToFile, readDataFromFile

partiesIndex = {'alice': 0, 'bob': 1}
partiesNames = {0: 'Alice', 1: 'Bob'}

class BellAtHome:
    seedsAB = np.array([0, 1])
    
    def __init__(self, n = 10000, epsilon = 0.2, k = 2, device = 'Alice'):
        '''
        Initialize a Bell@Home device
        
        Parameters
        ----------
        n : int, optional
            Number of rounds. The default is 10000.
        epsilon : real, optional
            Maximum bias allower per individual raw random bit. Between 0 and
            1. The default is 0.2.
        k : TYPE, optional
            Number of raw random bits used to create one question. The default
            is 1.
        device : string, optional
            Name of the device. Either 'Alice' or 'Bob'. The default is
            'Alice'.

        Returns
        -------
        None.

        '''
        self.n = int(n)
        self.epsilon = epsilon
        self.k = int(k)
        self.device = partiesNames[partiesIndex[device.lower()]]
        
        
        # Initialize the random seeds
        np.random.seed(self.seedsAB[0])
        seedsAlice = np.random.randint(1e9, size=(3))
        np.random.seed(self.seedsAB[1])
        seedsBob = np.random.randint(1e9, size=(3))
        self.seeds = {'Alice':seedsAlice, 'Bob':seedsBob}
        

    def removeAllData(self):
        '''
        Cleans the folder from all data files

        Returns
        -------
        None.

        '''
        for f in os.listdir('.'):
            if f.endswith('.dat'):
                os.remove(f)

    
    def targetSettings(self, device = None):
        '''
        The settings we want to bias towards
        
        Parameters
        ----------
        device : string, optional
            Name of the device. Either 'Alice' or 'Bob'. The default is own
            device name.

        Returns
        -------
        Array of Int64
            The target settings for the desired device.

        '''
        if device == None:
            device = self.device
        
        np.random.seed(self.seeds[device][0])
        return np.random.randint(2, size = self.n)
    
    
    def localStrategy(self):
        '''
        A table containing the answers to provide for each possible question
        a device might receive as a function of the hidden variable value

        Returns
        -------
        list
            The local strategies.

        '''
        if self.device == 'Alice':
            return [[0, 0], [0, 0], [0, 1], [0, 1]]
        else:
            return [[0, 0], [0, 1], [0, 0], [1, 0]]
    
    
    def hiddenVariable(self):
        '''
        The main hidden variable in the local model, telling which local
        strategy to use in each round

        Returns
        -------
        Array of Int64
            Values of the hidden variables for all rounds.

        '''
        targetAlice = self.targetSettings('Alice')
        targetBob = self.targetSettings('Bob')
        return targetAlice + 2*targetBob
    
    
    def additionalFlips(self):
        np.random.seed(2)
        return np.random.randint(2, size = self.n)
    
    
    def generateRandomness(self):
        '''
        Create the initial raw randomness file

        Returns
        -------
        None.

        '''
        # Initialize the random seed
        np.random.seed(self.seeds[self.device][1])
        
        X0 = np.random.randint(2, size=(self.k*self.n))
        
        writeDataToFile(X0, 'randomness' + self.device + '.dat')


    def computeQuestions(self):
        '''
        Computes questions using available randomness file

        Returns
        -------
        None.

        '''
        
        X = readDataFromFile('randomness' + self.device + '.dat')
        assert(X.size == self.k*self.n)
        X = X.reshape((self.n, self.k)).transpose()
        
        targetX = self.targetSettings()
        
        np.random.seed(self.seeds[self.device][2])
        
        runningValue = int(0)
        settings = np.zeros(self.n, dtype=int)
        for j in range(self.n):
            for i in range(self.k):
                if np.mod(runningValue + X[i,j], 2) != targetX[j]:
                    if np.random.sample() < self.epsilon:
                        X[i,j] = 1 - X[i,j]
                runningValue = np.mod(runningValue + X[i,j], 2)
            settings[j] = runningValue


        X = X.transpose().reshape(self.k*self.n)
        writeDataToFile(X, 'randomness' + self.device + '_biased.dat')
        writeDataToFile(settings, 'questions' + self.device + '.dat')
    
    
    def computeAverageBias(self):
        '''
        Evaluate the actual bias by comparing the raw randomness before and 
        after biasing.
        
        Returns
        -------
        double
            The observed bias.
        '''
        X0 = readDataFromFile('randomness' + self.device + '.dat')
        X = readDataFromFile('randomness' + self.device + '_biased.dat')
        assert(X0.size == self.k*self.n)
        assert(X.size == self.k*self.n)
        
        return 2*abs(X-X0).mean()
    
    
    def answerQuestions(self):
        '''
        Produce an answer for all questions

        Returns
        -------
        None.

        '''
        X = readDataFromFile('questions' + self.device + '.dat')
        assert(X.size == self.n)
        
        lamb = self.hiddenVariable()
        mu = self.additionalFlips()
        
        strategy = self.localStrategy()
        
        outcomes = np.zeros(self.n, dtype=int)
        for i in range(self.n):
            outcomes[i] = np.mod(strategy[lamb[i]][X[i]] + mu[i], 2)
        
        writeDataToFile(outcomes, 'answers' + self.device + '.dat')
    
    
    def computeCHSH(self):
        '''
        Computes the CHSH value from the questions and answers of both devices

        Returns
        -------
        double
            CHSH score

        '''
        X = readDataFromFile('questionsAlice.dat')
        Y = readDataFromFile('questionsBob.dat')
        A = readDataFromFile('answersAlice.dat')
        B = readDataFromFile('answersBob.dat')
        assert(X.size == self.n)
        assert(Y.size == self.n)
        assert(A.size == self.n)
        assert(B.size == self.n)
        
        w = sum((1+(-1)**(A+B+X*Y))/2)/self.n
        return 8*w-4


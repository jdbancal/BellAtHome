#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:26:00 2020

@author: bancal

This file provides some basic file writing and reading methods
"""

import numpy as np


# This function writes a string of binary variables into a file. 
# It breaks lines after every 64 characters for a better readability.
def writeDataToFile(data, fileName):
    text_file = open(fileName, "wt")
    pos = 0
    while len(data) > pos:
        nbToWrite = min(64, len(data)-pos)
        oneLine = ''.join([str(x) for x in data[pos:pos+nbToWrite]])
        assert (len(oneLine) == nbToWrite), 'More than one character needed to encode values in ' + oneLine
        text_file.write(oneLine+'\n')
        pos = pos + nbToWrite
    text_file.close()

# This function reads a string from a file, removing newline characters
def readDataFromFile(fileName):
    text_file = open(fileName)
    content = ''.join(text_file.read().splitlines())
    data = np.array([char for char in content], dtype=int)
    text_file.close()
    return data



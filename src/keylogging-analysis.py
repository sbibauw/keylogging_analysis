#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: keylogging_analysis.py
Author: Kobe Thonissen
Date: 2025-09-03
Version: 0.0.1
Description: definition of class KeyLoggingDataFrame for preprocessing and analysing keylogging data collected in the platforms Language Hero and Language Lab
"""

# structure of class methods
# 1. preprocessing methods

import pandas as pd


class KeyLoggingDataFrame():
    def __init__(self, path_keys, path_messages):
        """Initialize the KeyLoggingDataFrame with key and message data.
        input:
            path_keys: str, path to the keys/events logging data CSV file
            path_messages: str, path to the message data CSV file
        output: a KeyLoggingDataFrame object containing the merged csv files
        """
        # 1. open files
        try:
            keys = pd.read_csv(path_keys)
            messages = pd.read_csv(path_messages)
        except Exception as e:
            print(f"Error reading CSV files: {e}")

        #verify whether ids are unique
        



        # 2. merge dataframes
        self.data = pd.merge(keys, messages, on="message_id")

    def get_data(self):
        return self.data

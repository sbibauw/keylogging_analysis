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
    def __init__(self, path_keys:str, path_messages:str, system:str):
        """Initialize the KeyLoggingDataFrame with key and message data.
        input:
            path_keys: str, path to the keys/events logging data CSV file
            path_messages: str, path to the message data CSV file
            system: "lh" for Language Hero data or "ll" for Language Lab data
        output: a KeyLoggingDataFrame object containing the merged csv files
        """
        # 1. verify parameters
        if system not in ("lh", "ll"):
            raise ValueError(f"Invalid system '{system}'. Must be 'lh' or 'll'.")

        # 2. open files
        try:
            keys = pd.read_csv(path_keys)
            messages = pd.read_csv(path_messages)
        except Exception as e:
            raise FileNotFoundError(f"Error reading files: {e}")

        # 2. normalise column names
        if system == "lh":
            keys = keys.rename(columns={
                "id": "key_id",
                "content": "content",
                "eventForMessageType": "event_for_message_type",
                "moment": "event_time",
            })
            messages = messages.rename(columns={
                "id": "message_id",
                "MESSAGE_TYPE": "user_status",
                "transcript": "message_content",
                "creationDate": "created_at",
                "DIALOGUE_ID": "session_id"
            })
        elif system == "ll":
            keys = keys.rename(columns={
                "id": "key_id",
                "message": "content",
                "date": "moment",
                "message_id": "message_id"
            })
            messages = messages.rename(columns={
                "message_id": "message_uid",
                "id": "message_id",
                "content": "message_content",
                "created_at": "created_at",
                "session_id": "session_id"
            })

        # 3. read columns in correct format
        for col in ["key_id", "message_id", "session_id"]:
            if col in keys.columns:
                keys[col] = keys[col].astype(str)
            if col in messages.columns:
                messages[col] = messages[col].astype(str)
        for col in ["created_at", "moment"]:
            if col in keys.columns:
                keys[col] = pd.to_datetime(keys[col], errors='coerce')
            if col in messages.columns:
                messages[col] = pd.to_datetime(messages[col], errors='coerce')

        
        # 2. merge dataframes
        self.data = pd.merge(keys, messages, on="message_id")

    def get_data(self):
        return self.data

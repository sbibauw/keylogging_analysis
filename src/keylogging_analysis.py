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
import warnings


class KeyLoggingDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        # Initialize as an empty DataFrame
        super().__init__(*args, **kwargs)
        if self.empty:
            self._metadata = []


    def load_data_from_path(self, path_keys:str, path_messages:str, system:str):
        """Initialize the KeyLoggingDataFrame with key and message data.
        input:
            path_keys: str, path to the keys/events logging data CSV file
            path_messages: str, path to the message data CSV file
            system: "lh" for Language Hero data or "ll" for Language Lab data
        output: a KeyLoggingDataFrame object containing the merged csv files
        """
        # 1. verify parameters
        if system not in ("lh", "ll"):
            raise ValueError(f"Invalid system '{system}'. Must be 'lh' (Language Hero) or 'll' (Language Lab).")

        # 2. open files
        try:
            keys = pd.read_csv(path_keys)
            messages = pd.read_csv(path_messages)
        except Exception as e:
            raise FileNotFoundError(f"Error reading files: {e}")

        # 2. normalise column names
        try:
            if system == "lh":
                keys = keys.rename(columns={
                    "id": "key_id",
                    "content": "content",
                    "eventForMessageType": "event_for_message_type",
                    "moment": "key_time",
                    "MESSAGE_ID": "message_id"
                })
                messages = messages.rename(columns={
                    "id": "message_id",
                    "MESSAGE_TYPE": "user_status",
                    "transcript": "message_content",
                    "creationDate": "message_created_at",
                    "DIALOGUE_ID": "session_id"
                })
            elif system == "ll":
                keys = keys.rename(columns={
                    "id": "key_id",
                    "message": "content",
                    "date": "key_time",
                    "message_id": "message_id"
                })
                messages = messages.rename(columns={
                    "message_id": "message_uid",
                    "id": "message_id",
                    "content": "message_content",
                    "created_at": "message_created_at",
                    "session_id": "session_id"
                })
        except Exception as e:
            raise ValueError(f"Error renaming columns: {e}")

        # 3. read columns in correct format
        for col in ["key_id", "message_id", "session_id"]:
            if col in keys.columns:
                keys[col] = keys[col].astype(str)
            if col in messages.columns:
                messages[col] = messages[col].astype(str)

        if system == "ll":
            keys["key_time"] = pd.to_datetime(keys["key_time"], origin='unix', unit='ms', errors='coerce')
        for col in ["created_at", "key_time"]:
            if col in keys.columns:
                keys[col] = pd.to_datetime(keys[col], errors='coerce')
            if col in messages.columns:
                messages[col] = pd.to_datetime(messages[col], errors='coerce')

        # verify whether ids are unique
        if keys["key_id"].nunique() != len(keys):
            num_dropped = len(keys) - keys["key_id"].nunique()
            warnings.warn(f"Dropped {num_dropped} duplicate rows from keys data.")
            keys = keys.loc[keys["key_id"].drop_duplicates().index]

        if messages["message_id"].nunique() != len(messages):
            num_dropped = len(messages) - messages["message_id"].nunique()
            warnings.warn(f"Dropped {num_dropped} duplicate rows from messages data.")
            messages = messages.loc[messages["message_id"].drop_duplicates().index]

        # raise a warning if there are keys without a corresponding message
        if not keys["message_id"].isin(messages["message_id"]).all():
            num_orphaned = (~keys["message_id"].isin(messages["message_id"])).sum()
            warnings.warn(f"There are {num_orphaned} keys without a corresponding message.")

        # merge datadrames on message_id
        merged_df = pd.merge(keys, messages, on="message_id", how="inner")
        self.__init__(merged_df)



   

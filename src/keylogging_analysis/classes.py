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

import json
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from . import help_functions as hf

_DATA_DIR = Path(__file__).resolve().parent / "data"

class KeyLoggingDataFrame(pd.DataFrame):
    _metadata = ['system']

    def __init__(self, *args, **kwargs):
        prev_system = getattr(self, 'system', None)
        super().__init__(*args, **kwargs)
        if getattr(self, 'system', None) is None:
            self.system = prev_system

    @property
    def _constructor(self): # return KeyLoggingDataFrame when making copy
        return KeyLoggingDataFrame

    def _merge_and_update(self, df, colnames):
        """Merge computed columns back into self on key_id and reinitialize in-place."""
        if isinstance(colnames, str):
            colnames = [colnames]
        merged_df = self.merge(df[['key_id'] + colnames], on='key_id', how='left')
        self.__init__(merged_df)

    def _drop_anonymized(self):
        before = len(self)
        anon_mask = self['message_content'].str.contains('■', na=False)
        indices_to_drop = self[anon_mask].index
        self.drop(indices_to_drop, inplace=True)
        after = len(self)
        print(f"Dropped {before - after} events occurring in anonymized messages.")
          
    def _drop_nonsense(self):
        with open(_DATA_DIR / "lh_nonsense_message_ids.json") as f:
            message_ids = json.load(f)
        before = len(self)
        idx_to_drop = self[self["message_id"].isin(message_ids)].index
        self.drop(idx_to_drop, inplace=True)
        after = len(self)
        print(f"Dropped {before - after} events occurring in messages which contain nonsense bursts")

    def _drop_native(self):
        # TODO: verify that these IDs are correct for native speakers (currently same as nonsense list)
        with open(_DATA_DIR / "lh_native_message_ids.json") as f:
            message_ids = json.load(f)
        before = len(self)
        idx_to_drop = self[self["message_id"].isin(message_ids)].index
        self.drop(idx_to_drop, inplace=True)
        after = len(self)
        print(f"Dropped {before - after} events of native speakers")


    def load_default_data(self, system, include_anonymized:bool|None = None, include_nonsense:bool|None = None, include_native:bool|None = None) -> "KeyLoggingDataFrame":
        # 1. verify parameters
        if system not in ("lh", "ll"):
            raise ValueError(f"Invalid system '{system}'. Must be 'lh' (Language Hero) or 'll' (Language Lab).")

        if system != "lh" and include_anonymized is not None:
            raise ValueError(
                "include_anonymized can only be specified when system='lh'."
            )
        if system != "lh" and include_nonsense is not None:
            raise ValueError(
                "include_nonsense can only be specified when system='lh'."
            )
        if system != "lh" and include_native is not None:
            raise ValueError(
                "include_native can only be specified when system='lh'."
            )
        self.system = system
        
        # 2. load data
        base_dir = Path(__file__).resolve().parent
        if system == "lh":
            dtype_dict = {"key_id": int, "content": str, "event_for_message_type": str, "key_time": str, "message_id": int, "user_status": str, "message_content": str, "message_created_at": str, "session_id": int, "PERSONA_ID": float, "USER_ID": float, "TASK_ID": float, "application": str, "creationDate_conversation": str, "targetLang": int, "tutorLang": int, "hasTasks": bool, "CREATOR_ID": int, "SCENARIO_ID": int, "preCreated": bool, "simulated": bool, "archivedForProcessing": bool, "username": str}
            df = pd.read_csv(base_dir / "data/lh_default.csv", dtype=dtype_dict)
            df["key_time"] = pd.to_datetime(df["key_time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            df["message_created_at"] = pd.to_datetime(df["message_created_at"],format="%Y-%m-%d %H:%M:%S.%f", errors="coerce") 
            self.__init__(df)

        elif system == "ll":
            dtype_dict = {"key_id": int, "message_id": int, "content": str, "key_time": str, "message_uid": str, "message_content": str, "user_id": int, "session_id": int, "reply_to_message_id": str, "message_created_at": str}
            df = pd.read_csv(base_dir / "data/ll_default.csv", dtype=dtype_dict)
            df["key_time"] = pd.to_datetime(df["key_time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
            df["message_created_at"] = pd.to_datetime(df["message_created_at"], errors="coerce")
            self.__init__(df)

        print(f"Loaded {len(self)} events")

        # 3. Drop unwanted data
        if system == "lh":
            if not include_anonymized:
                self._drop_anonymized()
            if not include_nonsense:
                self._drop_nonsense()
            if not include_native:
                self._drop_native()
        print("KeyLoggingDataFrame of shape", self.shape)

        return self

    def load_data_from_path(self, system:str, path_keys:str, path_messages:str, include_anonymized:bool|None = None, include_nonsense:bool|None = None, include_native:bool|None = None) -> "KeyLoggingDataFrame":
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
        if system != "lh" and include_anonymized is not None:
            raise ValueError(
                "include_anonymized can only be specified when system='lh'."
            )
        if system != "lh" and include_nonsense is not None:
            raise ValueError(
                "include_nonsense can only be specified when system='lh'."
            )
        if system != "lh" and include_native is not None:
            raise ValueError(
                "include_native can only be specified when system='lh'."
            )
        self.system = system

        # 2. open files
        try:
            keys = pd.read_csv(path_keys)
            messages = pd.read_csv(path_messages)
        except Exception as e:
            raise FileNotFoundError(f"Error reading files: {e}")

        # 3. normalise column names
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

        # 4. add missing columns
        # Language Lab does not have event_for_message_type and user_status columns
        if system == "ll":
            if "event_for_message_type" not in keys.columns:
                keys["event_for_message_type"] = 0
                # set first event of each message to 5 if column content is empty
                first_events = keys.sort_values(by=["message_id", "key_time"]).groupby("message_id").head(1).index
                empty_content = keys[keys["content"].isnull()].index
                keys.loc[first_events.intersection(empty_content), "event_for_message_type"] = 5
            if "user_status" not in messages.columns:
                # TO DO: ideally user status already in messages
                messages["user_status"] = "L"
                messages.loc[messages["user_id"] == 8, "user_status"] = "T"



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

        # 4. Verify whether ids are unique
        if keys["key_id"].nunique() != len(keys):
            num_dropped = len(keys) - keys["key_id"].nunique()
            warnings.warn(f"Dropped {num_dropped} duplicate rows from keys data.")
            keys = keys.loc[keys["key_id"].drop_duplicates().index]

        if messages["message_id"].nunique() != len(messages):
            num_dropped = len(messages) - messages["message_id"].nunique()
            warnings.warn(f"Dropped {num_dropped} duplicate rows from messages data.")
            messages = messages.loc[messages["message_id"].drop_duplicates().index]

        # 5. Raise a warning if there are keys without a corresponding message
        if not keys["message_id"].isin(messages["message_id"]).all():
            num_orphaned = (~keys["message_id"].isin(messages["message_id"])).sum()
            warnings.warn(f"There are {num_orphaned} keys without a corresponding message. These will not be included in the final dataframe")

        # 4. Merge dataframes
        merged_df = pd.merge(keys, messages, on="message_id", how="inner")
        self.__init__(merged_df)

        print(f"Loaded {len(self)} events")

        # 5. Drop unwanted data
        if system == "lh":
            if not include_anonymized:
                self._drop_anonymized()
            if not include_nonsense:
                self._drop_nonsense()
            if not include_native:
                self._drop_native()
        print("Created KeyLoggingDataFrame of shape", self.shape)

        return self

    def add_iki(self, colname: str = "iki", include_first_key: bool = False, include_nontyping_events: bool = False) -> "KeyLoggingDataFrame":
        """Add inter-key interval (IKI) column to the dataframe. The IKI is the time difference in milliseconds between consecutive key presses within the same message.
        input: 
            colname: str, name of the new column to be added (default is 'iki'). Make sure it does not already exist in the dataframe.
        output: KeyLoggingDataFrame with new IKI column"""
        
        # 1. verify parameters
        if "key_time" not in self.columns:
            raise ValueError("DataFrame must contain 'key_time' column to calculate IKI.")
        if "message_id" not in self.columns:
            raise ValueError("DataFrame must contain 'message_id' column to calculate IKI.")
        if "event_for_message_type" not in self.columns:
            raise ValueError("ignore_nontyping_events can only be used if 'event_for_message_type' column is present")
        if colname in self.columns:
            raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")
        if include_nontyping_events is True and include_first_key is False:
            raise ValueError("include_first_key can only be False if include_nontyping_events is False")
        
        # 2. Make copy of dataframe in self
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # 3. Select events specified in parameters and compute IKI
        df["event_for_message_type"] = pd.to_numeric(df["event_for_message_type"])
        if not include_nontyping_events:
            if not include_first_key:
                df = df[df['event_for_message_type'] == 0]
            else:
                df = df[df['event_for_message_type'].isin([0, 5])]
        df['prev_key_time'] = df['key_time'].shift(1)
        df['prev_message_id'] = df['message_id'].shift(1)

        # Compute time difference in milliseconds
        time_diff_ms = (df['key_time'] - df['prev_key_time']).dt.total_seconds() * 1000

        if not include_nontyping_events:
            mask = (df['message_id'] == df['prev_message_id']) & (df['event_for_message_type'] == 0)
        else:
            mask = (df['message_id'] == df['prev_message_id'])
        df[colname] = np.where(mask, time_diff_ms, np.nan)

        # 4. Ensure numeric type and merge back
        df[colname] = pd.to_numeric(df[colname], errors='coerce')
        self._merge_and_update(df, colname)
        return self

    def add_pause(self, colname:str, method:str, threshold:float=None, a:float=None, iki_colname:str=None, include_first_key:bool=False, include_nontyping_events:bool=False) -> "KeyLoggingDataFrame":
        """Add a boolean pause column to the dataframe, which contains True when the key event is preceded by a pause and False if it is not. The IKI is set to NaN if there is no IKI available for the event
        Pauses can be either computed based on a fixed threshold IKI (in milliseconds) or individualized based on the median IKI of each user
        input:
            method: str, method to determine pauses: can be "fixed" or "individualized".
            threshold: float, IKI threshold in milliseconds above which a pause is marked.
            colname: str, name of the new column to be added (default is 'pause'). Make sure it does not already exist in the dataframe.
            include_first_key: bool, whether to include the first key event of each message when calculating IKIs (default is False).
            include_nontyping_events: bool, whether to include non-typing events (event_for_message_type != 0) when calculating IKIs (default is False).
        output: KeyLoggingDataFrame with new pause column"""
        # 1. verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. load data first throught load_default_data or load_data_from_path")
        if not method in ["fixed", "individualized"]:
            raise ValueError("Only 'fixed' and 'individualized' methods are supported.")
        if method == "fixed":
            if a is not None:
                raise AttributeError("Parameter 'a' is only used with method='individualized'")
            if threshold is None:
                threshold = 200
                print("No threshold specified, using default of 200 ms")
            if not isinstance(threshold, (int, float)) or threshold <= 0:
                raise ValueError("Threshold must be a positive number.")
        elif method == "individualized":
            if threshold is not None:
                raise AttributeError("Parameter 'threshold' is only used with method='fixed'")
            if a is None:
                a = 2
                print("No 'a' value specified, using default of 2")
            if not isinstance(a, (int, float)):
                raise ValueError("Parameter 'a' must be a number.")
        if "key_time" not in self.columns:
            raise ValueError("DataFrame must contain 'key_time' column to calculate pauses.")
        if "message_id" not in self.columns:
            raise ValueError("DataFrame must contain 'message_id' column to calculate pauses.")
        if "event_for_message_type" not in self.columns:
            raise ValueError("ignore_nontyping_events can only be used if 'event_for_message_type' column is present")
        if not colname:
            colname = hf.generate_colname("pause", self.columns)
        if colname in self.columns:
            raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")
        if include_nontyping_events is True and include_first_key is False:
            raise ValueError("include_first_key can only be False if include_nontyping_events is False")
        if iki_colname and iki_colname not in self.columns:
                raise ValueError(f"IKI column '{iki_colname}' not found in DataFrame.")
                
        # 2. Make copy of dataframe in self
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # 3. add IKI column if not already present
        # indicate pauses
        if method == "fixed":
            # select non nan IKI key_ids
            non_nan_iki = df[df[iki_colname].notna()].index
            df[colname] = np.nan
            df[colname] = df[colname].astype("boolean")
            df.loc[non_nan_iki, colname] = df.loc[non_nan_iki, iki_colname] > threshold
            
        elif method == "individualized":
            # get median IKI per user
            median_iki = df.groupby("user_id")[iki_colname].median()
            median_mad = df.groupby("user_id")[iki_colname].apply(lambda x: (x - x.median()).abs().median())
            # addcolumn with median IKI per user
            df = df.merge(median_iki.rename("median_iki"), on="user_id", how="left")
            df = df.merge(median_mad.rename("median_mad"), on="user_id", how="left")
            # select non nan IKI key_ids
            non_nan_iki = df[df[iki_colname].notna()].index
            df.loc[non_nan_iki, "individualized_threshold"] = df.loc[non_nan_iki, "median_iki"] + a * df.loc[non_nan_iki, "median_mad"]
            df[colname] = np.nan
            df[colname] = df[colname].astype("boolean")
            df.loc[non_nan_iki, colname] = df.loc[non_nan_iki, iki_colname] > df.loc[non_nan_iki, "individualized_threshold"]

        # 4. Merge pause column back into self on key_id
        self._merge_and_update(df, colname)
        return self

    def add_pburst(self, colname:str=None, pause_colname:str=None, pause_method:str=None, pause_threshold:float=None, pause_a:float=None, iki_colname:str=None) -> "KeyLoggingDataFrame":
        """Add a boolean pburst column to the dataframe, which contains True when the key event is part of a pause burst and False if it is not. A pause burst is defined as a sequence of at least two consecutive key events that are each preceded by a pause.
        input:
            colname: str, name of the new column to be added (default is 'pburst'). Make sure it does not already exist in the dataframe.
            pause_colname: str, name of the column containing pause information (default is 'pause').
        output: KeyLoggingDataFrame with new pburst column"""
        # 1. verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. Load data first throught load_default_data or load_data_from_path")
        if "key_time" not in self.columns:
            raise ValueError("DataFrame must contain 'key_time' column to calculate pbursts.")
        if "message_id" not in self.columns:
            raise ValueError("DataFrame must contain 'message_id' column to calculate pbursts.")
        if not pd.api.types.is_bool_dtype(self[pause_colname]) and not pd.api.types.is_integer_dtype(self[pause_colname]) and not pd.api.types.is_float_dtype(self[pause_colname]):
            raise ValueError(f"Pause column '{pause_colname}' must be of boolean or numeric dtype.")
        if not colname:
            colname = hf.generate_colname("pburst", self.columns)
        if colname in self.columns:
            raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")
            
                
        # 2. Make copy of dataframe in self
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # 3. create pause column if not already present
        #if pause_method is not None:
        #    if pause_colname is None:
        #        pause_colname = hf.generate_colname("pause", df.columns)
        #    df.add_pause(colname=pause_colname, method=pause_method, threshold=pause_threshold, a=pause_a, iki_colname=iki_colname, include_first_key=True, include_nontyping_events=False)

        # Create pburst column
        first_characters = df[df["event_for_message_type"] == 0].sort_values(by=["message_id", "key_time"]).groupby("message_id").head(1).index
        df["pburst"] = "O"  # default to 'O' (other/NaN)
        df.loc[df[pause_colname] == False, "pburst"] = "I"
        df.loc[df[pause_colname] == True, "pburst"] = "B"
        df.loc[first_characters, "pburst"] = "B"

        # merge df to self
        self._merge_and_update(df, 'pburst')
        return self

    def add_action(self, colname:str=None):
        """
        Add an action column to the dataframe, which contains the action performed with each key event. Actions can be "addition", "deletion", "modification" or "no change".
        """# 1. verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. load data first throught load_default_data or load_data_from_path")
        if "event_for_message_type" not in self.columns:
            raise ValueError("DataFrame must contain 'event_for_message_type' column to calculate actions.")
        if "content" not in self.columns:
            raise ValueError("DataFrame must contain 'content' column to calculate actions.")
        if not colname:
            colname = hf.generate_colname("action", self.columns)
        elif colname in self.columns:
            raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")
            
        # 2. copy df
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # 3. select edits
        df["event_for_message_type"] = pd.to_numeric(df["event_for_message_type"])
        df = df[df['event_for_message_type'] == 0]
    
        # add a columns "previous_content", which contains the content of the previous event of the message. Contains an empty string if the event is the first event of the message
        df["previous_content"] = df["content"].shift(1)
        df["previous_message_id"] = df["message_id"].shift(1)

        # add action using pandas apply
        df['action'] = df.apply(
            lambda row: hf.detect_action(row['content'], row['previous_content']),
            axis=1
        )

        # set action to NaN if different messages
        df.loc[df['message_id'] != df['previous_message_id'], 'action'] = None


        # 4. Merge action column back into self on key_id
        self._merge_and_update(df, 'action')
        return self
        
    def add_span(self, colnames:list=[None, None, None, None], selection:list=[True,True,True,True]):
        """
        Add 4 columns to the dataframe which indicate the range of the event: start_deletion_span, end_deletion_span, start_addition_span, end_addition_span
        """
        # 1. verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. load data first throught load_default_data or load_data_from_path")
        if "event_for_message_type" not in self.columns:
            raise ValueError("DataFrame must contain 'action' column to calculate ranges. Use add_action() to add this column.")
        if "content" not in self.columns:
            raise ValueError("DataFrame must contain 'content' column to calculate ranges.")
        if not colnames or colnames == [None, None, None, None]:
            colnames = []
            for base in ["start_deletion_span", "end_deletion_span", "start_addition_span", "end_addition_span"]:
                colnames.append(hf.generate_colname(base, self.columns))
        elif not isinstance(colnames, list) or len(colnames) != 4:
            raise ValueError("colnames must be a list of 4 strings (or None).")
        else:
            for colname in colnames:
                if colname in self.columns:
                    raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")
                    

        # 2. copy df
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # 3. select edits only
        df["event_for_message_type"] = pd.to_numeric(df["event_for_message_type"])
        df = df[df['event_for_message_type'] == 0]
    
        # add a columns "previous_content", which contains the content of the previous event of the message. Contains an empty string if the event is the first event of the message
        df["previous_content"] = df["content"].shift(1)
        df["previous_message_id"] = df["message_id"].shift(1)

        # add range using pandas apply
        if selection[0] == True:
            if not colnames[0]:
                colnames[0] = hf.generate_colname("start_deletion_span", df.columns)
            df[colnames[0]] = df.apply(
                lambda row: hf.detect_start_deletion_span(row['content'], row['previous_content']),
                axis=1
            )
            
        if selection[1] == True:
            if not colnames[1]:
                colnames[1] = hf.generate_colname("end_deletion_span", df.columns)
            df[colnames[1]] = df.apply(
                lambda row: hf.detect_end_deletion_span(row['content'], row['previous_content']),
                axis=1
            )
        if selection[2] == True:
            if not colnames[2]:
                colnames[2] = hf.generate_colname("start_addition_span", df.columns)
            df[colnames[2]] = df.apply(
                lambda row: hf.detect_start_addition_span(row['content'], row['previous_content']),
                axis=1
            )

        if selection[3] == True:
            if not colnames[3]:
                colnames[3] = hf.generate_colname("end_addition_span", df.columns)
            df[colnames[3]] = df.apply(
                lambda row: hf.detect_end_addition_span(row['content'], row['previous_content']),
                axis=1
            )

        added_colnames = [col for col, sel in zip(colnames, selection) if sel]
        # set spans to NaN if different messages
        df.loc[df['message_id'] != df['previous_message_id'], added_colnames] = None

        # 4. Merge span columns back into self on key_id
        self._merge_and_update(df, added_colnames)
        return self

    def add_length(self, colnames:list=[None, None], selection=[True, True]) -> "KeyLoggingDataFrame":
        """
        Add 2 columns to the dataframe which indicate the length of the the deleted sequence and the length of the added sequence
        """
        # 1. verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. load data first throught load_default_data or load_data_from_path")
        if "event_for_message_type" not in self.columns:
            raise ValueError("DataFrame must contain 'event_for_message_type' column to calculate lengths.")
        if "content" not in self.columns:
            raise ValueError("DataFrame must contain 'content' column to calculate lengths.")
        if not colnames or colnames == [None, None]:
            colnames = [
                hf.generate_colname("length_deletion", self.columns) if selection[0] else None,
                hf.generate_colname("length_addition", self.columns) if selection[1] else None,
            ]
        elif not isinstance(colnames, list) or len(colnames) != 2:
            raise ValueError("colnames must be a list of 2 strings (or None).")
        else:
            for colname in colnames:
                if colname in self.columns:
                    raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")
                
        # 2. copy df
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # 3. select edits only
        df["event_for_message_type"] = pd.to_numeric(df["event_for_message_type"])
        df = df[df['event_for_message_type'] == 0]

        #
        if selection[0]: # deletion length
            # get deletion span
            startcol = hf.generate_colname("start_deletion_span", df.columns)
            endcol = hf.generate_colname("end_deletion_span", df.columns)
            df.add_span(colnames=[startcol, endcol, None, None], selection=[True, True, False, False])
            # calculate length
            df[colnames[0]] = df[endcol] - df[startcol]
        
        if selection[1]: # addition length
            # get addition span
            startcol = hf.generate_colname("start_addition_span", df.columns)
            endcol = hf.generate_colname("end_addition_span", df.columns)
            df.add_span(colnames=[None, None, startcol, endcol], selection=[False, False, True, True])
            # calculate length
            df[colnames[1]] = df[endcol] - df[startcol]

        
        # 4. Merge length columns back into self on key_id
        added_colnames = [col for col, sel in zip(colnames, selection) if sel]
        self._merge_and_update(df, added_colnames)
        return self
    

    def add_rburst(self, colname:str=None, action_colname:str=None) -> "KeyLoggingDataFrame":
        """Add a column to the dataframe annotating in IOB format whether the key event is the beginning of a revision burst (B), inside a revision burst (I) or outside a revision burst (O).
        A revision burst is a sequence of consecutive non-addition events (deletions, substitutions)."""
        # 1. verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. Load data first through load_default_data or load_data_from_path")
        if not action_colname:
            raise ValueError("Action column name must be specified.")
        if action_colname not in self.columns:
            raise ValueError(f"Action column '{action_colname}' not found in DataFrame.")
        if "message_id" not in self.columns:
            raise ValueError("DataFrame must contain 'message_id' column to calculate rbursts.")
        if not colname:
            colname = hf.generate_colname("rburst", self.columns)
        if colname in self.columns:
            raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")

        # 2. Make copy of dataframe in self
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # 3. Create rburst column based on action column
        # A revision is any non-addition action (deletion, substitution)
        is_revision = df[action_colname].isin(["deletion", "substitution"])
        prev_is_revision = is_revision.shift(1, fill_value=False)
        prev_message_id = df["message_id"].shift(1)
        same_message = df["message_id"] == prev_message_id

        df[colname] = "O"  # default to outside
        # B = start of revision burst (revision but previous was not, or different message)
        df.loc[is_revision & (~prev_is_revision | ~same_message), colname] = "B"
        # I = continuation of revision burst
        df.loc[is_revision & prev_is_revision & same_message, colname] = "I"

        # merge df to self
        self._merge_and_update(df, colname)
        return self

    def add_distance_to_end(self, colname:str=None) -> "KeyLoggingDataFrame":
        # verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. load data first throught load_default_data or load_data_from_path")
        if "content" not in self.columns:
            raise ValueError("DataFrame must contain 'content' column to calculate distance to end.")
        if "key_id" not in self.columns:
            raise ValueError("DataFrame must contain 'key_id' column to calculate distance to end.")
            
        df = self.copy()
        end_del_colname = hf.generate_colname("end_deletion_span", df.columns)
        end_add_colname  = hf.generate_colname("end_addition_span", df.columns)
        df = df.add_span(colnames=[None, end_del_colname, None, end_add_colname], selection=[False,True,False,True])
        df[colname] = df.apply(
            lambda row: hf.detect_distance_to_end(row['content'], row[end_del_colname], row[end_add_colname]),
            axis=1
        )
        self._merge_and_update(df, colname)
        return self

    def add_revision(self, colname:str=None, action_colname:str=None):
        """add a column indicating which events are part of a revision according to the IOB format"""
        # 1. verify parameters
        if self.empty:
            raise ValueError("DataFrame is empty. load data first throught load_default_data or load_data_from_path")
        if not action_colname:
            raise ValueError("Action column name must be specified.")
        if action_colname not in self.columns:
            raise ValueError(f"Action column '{action_colname}' not found in DataFrame.")
        if "message_id" not in self.columns:
            raise ValueError("DataFrame must contain 'message_id' column to calculate revisions.")
        if "key_id" not in self.columns:
            raise ValueError("DataFrame must contain 'key_id' column to calculate revisions.")
        if not pd.api.types.is_object_dtype(self[action_colname]) and not pd.api.types.is_string_dtype(self[action_colname]):
            raise ValueError(f"Action column '{action_colname}' must be of string dtype.")
        if not set(self[action_colname].dropna().unique()).issubset({"addition", "deletion", "modification", "no change"}):
            raise ValueError(f"Action column '{action_colname}' must only contain the values 'addition', 'deletion', 'modification' and 'no change'.")
        if colname in self.columns:
            raise ValueError(f"Column '{colname}' already exists in the DataFrame. Please choose a different name.")
        if not colname:
            colname = hf.generate_colname("revision", self.columns)

        # 2. Make copy of dataframe in self
        df = self.copy()
        df = df.sort_values(by=['message_id', 'key_time'])

        # add action and distance_to_end


        # 3. Create revision column
        df["revision"] = "O"

        # Mark deletions and within-text insertions and  as 'I'
            

        first_characters = df[df["event_for_message_type"] == 0].sort_values(by=["message_id", "key_time"]).groupby("message_id").head(1).index
        df["revision"] = "O"  # default to 'O' (other/NaN)
        df.loc[df[action_colname] != "no change", "revision"] = "B"
        df.loc[first_characters, "revision"] = "O"

        # merge df to self
        self._merge_and_update(df, 'revision')
        return self


    def pause_dataframe(self, pause_colname:str=None, iki_colname:str=None) -> pd.DataFrame:
        """returns a Pandas DataFrame in which each row represents a pause in the KeyLoggingDataFrame. Besides the normal keylogging data, the dataframe contains the following columns:
        - pause_location: "word_start", "word_middle", "word_end" (words are defined as sequences of alphannumeric characters)
        """
        raise NotImplementedError("pause_dataframe is not yet implemented.")

    def burst_dataframe(self, burst_colname:str=None, iki_colname:str=None, action_colname:str=None) -> pd.DataFrame:
        """Returns a dataframe containing the character length and duration of each burst in the KeyLoggingDataFrame.
        input: self is a KeyLoggingDataFrame 
            burst_colname: str, name of the column containing burst tags in IOB format (generated with add_pburst or add_rburst)
        output: pd.DataFrame with one row per burst and columns: burst_id, message_id, user_id, session_id, start_time, end_time, duration, length
        """
        # verify parameters
        if not burst_colname:
            raise ValueError("Burst column name must be specified.")
        if burst_colname not in self.columns:
            raise ValueError(f"Burst column '{burst_colname}' not found in DataFrame.")
        if not iki_colname:
            raise ValueError("IKI column name must be specified.")
        if iki_colname not in self.columns:
            raise ValueError(f"IKI column '{iki_colname}' not found in DataFrame.")
        if self.empty:
            return pd.DataFrame()  # return empty dataframe if self is empty

        events_df = self.copy()
        # group by message id and apply get burst_metrics to each group
        burst_df = events_df.groupby("message_id").apply(
            lambda x: hf.get_burst_metrics(x, burst_colname=burst_colname, action_colname=action_colname, iki_colname=iki_colname)
        ).reset_index(drop=True)
        return burst_df

    def revision_dataframe(self, rburst_colname:str=None, action_colname:str=None) -> pd.DataFrame:
        raise NotImplementedError("revision_dataframe is not yet implemented.")

    def pburst_analysis(self, pburst_colname:str=None) -> "KeyLoggingDataFrame":
        raise NotImplementedError("pburst_analysis is not yet implemented.")

    def pause_analysis(self, pause_colname:str=None, iki_colname:str=None) -> pd.DataFrame:
        # verify parameters
        if not pause_colname:
            pause_colname = hf.generate_colname("pause", self.columns)
        if pause_colname not in self.columns:
            raise ValueError(f"Pause column '{pause_colname}' not found in DataFrame.")
        if not pd.api.types.is_bool_dtype(self[pause_colname]) and not pd.api.types.is_integer_dtype(self[pause_colname]) and not pd.api.types.is_float_dtype(self[pause_colname]):
            raise ValueError(f"Pause column '{pause_colname}' must be of boolean or numeric dtype.")

        df = self.copy()
        df["event_for_message_type"] = pd.to_numeric(df["event_for_message_type"])
        df["iki"] = pd.to_numeric(df["iki"], errors='coerce')
        df = df[df['event_for_message_type'] == 0]  # consider only typing events

        # calculate summary statistics
        nb_keys = len(df)
        nb_pauses = df[pause_colname].sum()
        percentage_pauses_per_keys = nb_pauses / nb_keys * 100 if nb_keys > 0 else None
        total_duration = df[iki_colname].sum()
        pause_duration = df.loc[df[pause_colname] == True, iki_colname].sum()
        percentage_pause_duration = pause_duration / total_duration * 100 if total_duration > 0 else None
        avg_pause_length = df.loc[df[pause_colname] == True, iki_colname].mean()
        std_pause_length = df.loc[df[pause_colname] == True, iki_colname].std()
        median_pause_length = df.loc[df[pause_colname] == True, iki_colname].median()
        mad_pause_length = (df.loc[df[pause_colname] == True, iki_colname] - df.loc[df[pause_colname] == True, iki_colname].median()).abs().median()
        max_pause_length = df.loc[df[pause_colname] == True, iki_colname].max()
        min_pause_length = df.loc[df[pause_colname] == True, iki_colname].min()

        # create summary dataframe
        metrics = {
            "nb_pauses": int(nb_pauses) if pd.notna(nb_pauses) else None,
            "percentage_pauses_per_keys": round(percentage_pauses_per_keys,3) if pd.notna(percentage_pauses_per_keys) else None,
            "total_pause_duration": int(pause_duration) if pd.notna(pause_duration) else None,
            "percentage_pause_duration": round(percentage_pause_duration,3) if pd.notna(percentage_pause_duration) else None,
            "mean_pause_length": round(avg_pause_length,3) if pd.notna(avg_pause_length) else None,
            "std_pause_length": round(std_pause_length,3) if pd.notna(std_pause_length) else None,
            "median_pause_length": int(median_pause_length) if pd.notna(median_pause_length) else None,
            "mad_pause_length": int(mad_pause_length) if pd.notna(mad_pause_length) else None,
            "max_pause_length": int(max_pause_length) if pd.notna(max_pause_length) else None,
            "min_pause_length": int(min_pause_length) if pd.notna(min_pause_length) else None
        }
        metrics = pd.DataFrame([metrics])
        return metrics
        
    
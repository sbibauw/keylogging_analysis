# About `keylogging_analysis`

`keylogging_analysis` is a Python module aimed at processsing and analysing data from the **Language Hero** ("lh") and **Language Lab** ("ll") datasets. It is useful to researchers interested in **L2 language fluency**, **chatbot-student interactions** or **keylogging data**.

# How to install?

(pip)

# On Language Hero and Language Lab data

The Language Hero corpus contains keylogs of L2 learner interactions with conversational agents within the educational game Language Hero, developed by Linguineo. 

The Language Lab data contains keylogs of L2 learners and their tutors within a chat environment.

# Use of `keylogging_analysis`

The `keylogging_analysis` package facilitates analysis of keylogging data through the class KeyLoggingDataFrame, conceived as a subclass of a Pandas DataFrame. Besides the traditional methods applicable to Pandas dataframes, the class allows for additional methods aimed at loading keylogging data, processing it, computing metrics on the writing process and converting the data into KeyLog format. This section explains step by step how to proceed with the analysis. A more detailed description of the methods and their requirements can be found in the documentation below.

## 1. import the class and initiate an empty instance of this class

First, import the class KeyLoggingDataframe and instantiate an empty instance (e.g. kldf) to store the data in.
```
from keylogging_analysis import KeyLoggingDataFrame

kldf = KeyLoggingDataFrame()
```
Or alternatively
```
import keylogging_analysis as ka

kldf = ka.KeyloggingDataFrame()
```

## loading and preprocessing keylogging data 

To load the default datasets of either Language Hero or Language Lab into the KeyLoggingDataFrame instance defined in the previous step, use the method `load_default_data()`. The parameter `system` specifies whether to load the default dataset of Language Hero (`"lh"`) or Language Lab (`"ll"`)

```
kldf.load_default_data(system="lh")
```

Alternatively, the method `load_data_from_path()` allows other datasets to be uploaded to the KeyLoggingDataFrame instance, through specifying the absolute or relative path towards the .csv files containing the keys (`path_keys`) and the messages (`path_messages`). This may be useful when analysing a selection or a preprocessed version of the corpora. The parameter `system` is still required to specify to which column names to expect. Make sure the specified dataframes respect the constraints detailed in the documentation below.

```
kldf.load_data_from_path(system="ll", path_keys="keys.csv", path_messages="messages.csv")
```

For Language Hero data only (`system="lh"`), `load_default_data` and `load_data_from_path` drop by default events occurring in anonymized messages, nonsense messages (which contain keysmashes or character repetitions) and messages from native speakers. This behaviour can be undone by setting the parameters `include_anonymized`, `include_nonsense` or `include_native` to `True`.

```
kldf.load_data_from_path(system="ll", path_keys="keys.csv", path_messages="messages.csv", include_anonymized=False, include_nonsense=True, include_native=True)
```

## Analysis

Besides the standard methods of pandas DataFrames, the class KeyLoggingDataFrames allows for specific methods related to writing process analysis to be performed on the data. The methods `add_iki()`, `add_pause()`, `add_action()` add a column to the original dataframe, which can be analysed later. Mek sure to specify column names that don't exist in the original column yet.
```
kldf.add_iki(colname="iki")
print(kldf[["content", "event_for_message_type", "iki"]].head())
_____________________________________________________________________________________________________________________________________________
  content  event_for_message_type   iki
0       H                       0    NaN  
1      He                       0  123.0 
2     Hel                       0  110.0  
3    Hell                       0  145.0  
4   Hello                       0  145.0 
```

# Documentation `keylogging_analysis version 0.0.1`

> ## KeyLoggingDataFrame.load_default_data(system, include_anonymized=None, include_nonsense=None, include_native:bool=None*)
> 
> Load the default dataset from Language Hero or Language Lab and preprocess it according to the parameters
> 
> ### input:
> 
>   **system :** {'lh' or 'll'}
> 
> Choose between Language Hero data ('lh') and ('ll')
>
>   **include_anonymized :** bool|None, default None
> 
> `system = 'lh'`: Sensitive information in the Language Hero data is anonymized as a sequence of '■' characters (Unicode U+25A0). If `include_anonymized` equals `False` or `None`, these data are dropped from the dataframe (default behavior)
> 
> `'system = 'll'` : include_anonymized` can only be `None`
>
>  **include_nonsense :** bool|None, default None
> 
> `system = 'lh'`: If `False` or `None`, keysmashes and character repetitions are dropped from the dataframe (default behavior)
> 
> `'system = 'll'` : include_nonsense` can only be `None`
>
>  **include_native :** bool|None, default None
> 
> `system = 'lh'`:  If `False` or `None`, keylogging data from native speakers are dropped from the dataframe (default behavior)
> 
> `'system = 'll'` : include_native` can only be `None`
>  
> ### output:
>
> dataframe of class KeyLoggingDataFrame (which supports the same functions as `pandas` dataframes, but with additional functions)



> ## KeyLoggingDataFrame.load_data_from_path(system, path_keys, path_messages, include_anonymized=None, include_nonsense=None, include_native:bool=None*)
> 
> Initialize the KeyLoggingDataFrame with keys data 
> 
> ### input:
> 
> **system :** {'lh' or 'll'}
> 
> Choose between Language Hero data ('lh') and ('ll'). This is required to make sure the files respect the format of the datasystems. At least the following column names are required in the provided dataframes (additional columns are allowed).
>
> - `system = 'lh'` :
>   - messages : "id", "MESSAGE_TYPE", "transcript", "creationDate", "DIALOGUE_ID"
>   - keys : "id", "content", "eventForMessageType", "moment", "MESSAGE_ID"
> - `system = 'll'` :
>   - messages : "message_id", "id", "content", "created_at", "session_id"
>   - keys : "id", "message", "date", "message_id"
>  
> **path_keys :** str
>
> Absolute or relative path to the keys file
>
> > **path_keys :** str
>
> Absolute or relative path to the messages file
> 
> **include_anonymized :** bool|None, default None
> 
> `system = 'lh'`: Sensitive information in the Language Hero data is anonymized as a sequence of '■' characters (Unicode U+25A0). If `include_anonymized` equals `False` or `None`, these data are dropped from the dataframe (default behavior)
> 
> `'system = 'll'` : include_anonymized` can only be `None`
>
>  **include_nonsense :** bool|None, default None
> 
> `system = 'lh'`: If `False` or `None`, keysmashes and character repetitions are dropped from the dataframe (default behavior)
> 
> `'system = 'll'` : include_nonsense` can only be `None`
>
>  **include_native :** bool|None, default None
> 
> `system = 'lh'`:  If `False` or `None`, keylogging data from native speakers are dropped from the dataframe (default behavior)
> 
> `'system = 'll'` : include_native` can only be `None`
>  
> ### output:
>
> dataframe of class KeyLoggingDataFrame (which supports the same functions as `pandas` dataframes, but with additional functions). The initial keys and mesages dataframes are merged on the message id column.



## KeyLoggingDataFrame.add_ikis(colname="iki", include_first_key=False, include_nontyping_events=False)

> Add a column containing the **Inter-Keystroke Interval (IKI)** to the dataframe. The IKI measures the time difference (in milliseconds) between consecutive key presses within the same message and is used by other methods for computing pauses and writing speed
>
> ### input:
>
> **colname :** str, default `"iki"`  
>
> Name of the new column to be added.  
> The column name must not already exist in the dataframe.  
>
> **include_first_key :** bool, default `False`  
>
> - If `False`: the first key of each message is excluded from IKI computation.  
> - If `True`: the IKI for the first event in a message is computed (the pause before writing).  
>
> **include_nontyping_events :** bool, default `False`  
>
> - If `False`: only typing events (`event_for_message_type == 0`) are used for IKI computation.  
> - If `True`: all events are considered when computing IKI (including help requests, consultation of feedback and message sending).  
>
> ⚠️ Constraint:  If `include_nontyping_events=True`, then `include_first_key` must also be `True`.  
>
>
> ### output:
>
> A `KeyLoggingDataFrame` (same class as the input) with an additional column containing IKI values:  

> - Each value corresponds to the time difference (in ms) between the current key event and the previous key event (within the same `message_id`).  
> - Non-applicable events are filled with `NaN`.


> ### kldf.add_pauses(threshold, name="pause")
> Adds a column "pause" to the merged frame, which contains a boolean value indicating whether the event was preceded by a pause or not. The method for defining a pause can be specified under

> ### kldf.add_p_bursts(method)
> Adds a column "p_burst"

> ### kldf.add_r_bursts()

> ### kldf.add_revisions(method)

> ### add_location

> ### kldf.writing_speed(method)
> Get the writing speed of a dataframe in characters per minute

> ### kldf.bursts()
> Get info on bursts (mean, median, list of lengths)????

> ### kldf.median_iki()
> nb bursts
>   fabra!!!
> ### kldf.revision_rate()

> ### kldf.process_characters()

> ### kldf.product_characters()

> ### kldf.product_process_ratio()

> ### kldf.pause_rate()

> ### kldf.pause_location()
> 

## convertion
> ### kldf.convert()

messages

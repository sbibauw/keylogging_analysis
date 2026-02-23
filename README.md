# About `keylogging_analysis`

`keylogging_analysis` is a Python module for processing and analyzing keylogging data from the **Language Hero** ("lh") and **Language Lab** ("ll") datasets. It is designed for researchers and educators interested in **L2 language fluency**, **chatbot-student interactions**, or **keystroke logging**. The package provides a user-friendly interface for loading, cleaning, and analyzing keylogging data, and includes advanced methods for extracting writing process metrics.

# Development

- Load basic dataframe
- Add IKI column
- To do:
  - [ ] ....
  - [ ] Check if _method_pandas still works in Pandas x.x.x



# How to install?

The packaged can be easily installed with the [pip package installer](https://pypi.org/project/pip/), using the following line in the terminal (both on Windows and on Mac). 

```
pip install keylogging_analysis
```

# On Language Hero and Language Lab data

The Language Hero corpus contains keylogs of L2 learner interactions with conversational agents within the educational game Language Hero, developed by Linguineo. 

The Language Lab data contains keylogs of L2 learners and their tutors within a chat environment.

# Use of `keylogging_analysis`

The `keylogging_analysis` package provides the `KeyLoggingDataFrame` class, which extends the functionality of a standard pandas DataFrame with specialized methods for keylogging data. You can use all regular pandas DataFrame operations, but also benefit from additional methods for loading, cleaning, and analyzing writing process data. The workflow is as follows:

1. **Import the class and create an instance**: Start by importing `KeyLoggingDataFrame` and creating an empty instance. This instance will hold your data.
2. **Load your data**: Use `load_default_data()` to load the built-in datasets, or `load_data_from_path()` to load your own CSV files. The `system` parameter specifies whether you are working with Language Hero (`"lh"`) or Language Lab (`"ll"`) data.
3. **Preprocess and clean**: The class provides methods to automatically remove anonymized, nonsense, or native-speaker data (for Language Hero), or you can keep all data by setting the appropriate parameters.
4. **Add columns containing information in the writing process**: Use methods like `add_iki()`, `add_pause()`, `add_pburst()`, `add_action()`, `add_span()`, and `add_length()` to compute writing process metrics and add them as new columns to your DataFrame. You can then use pandas or the provided analysis methods to further explore your data.
5. **Apply analysis methods to the added columns**: Once the required columns are added to the original dataframe, apply methods like `pause_analysis()` or `burst_analysis()` to extract writing process metrics or use `burst_dataframe()` or `revision_dataframe()` to generate dataframes in which each row represents a burst or revision respectively, which can be used for further analysis.
6. **Export or visualize**: After analysis, you can export your DataFrame to CSV or use pandas/plotting libraries for visualization.

Each method is designed to be easy to use, with sensible defaults and clear error messages. See the technical documentation below for details on each method and its parameters.


# Documentation `keylogging_analysis version 0.0.1`

## KeyLoggingDataFrame methods

### Data Loading and Preprocessing

#### `load_default_data(system, include_anonymized=None, include_nonsense=None, include_native=None)`
Loads and preprocesses the default dataset for the specified system ("lh" or "ll"). For Language Hero, you can exclude anonymized, nonsense, or native-speaker data using the optional parameters. Returns a KeyLoggingDataFrame with merged and cleaned data.

#### `load_data_from_path(system, path_keys, path_messages, include_anonymized=None, include_nonsense=None, include_native=None)`
Loads and merges key and message data from specified CSV files. The `system` parameter ensures the correct column mapping. Optional parameters for Language Hero allow filtering of anonymized, nonsense, or native-speaker data.

### Writing Process Metrics

#### `add_iki(colname="iki", include_first_key=False, include_nontyping_events=False)`
Adds a column with the Inter-Keystroke Interval (IKI) in milliseconds. IKI is the time between consecutive key events within the same message. If `include_first_key` is `True`, the first key of each message is included; otherwise, it is set to NaN. If `include_nontyping_events` is `True`, all events are considered; otherwise, only typing events (`event_for_message_type == 0`).

#### `add_pause(colname, method, threshold=None, a=None, iki_colname=None, include_first_key=False, include_nontyping_events=False)`
Adds a boolean column indicating whether a key event is preceded by a pause. Pauses can be defined by a fixed IKI threshold (`method="fixed"`, `threshold` in ms) or individualized per user (`method="individualized"`, using a multiple `a` of the user's median IKI). The pause is `True` if the IKI exceeds the threshold, `False` otherwise, and NaN if IKI is not available.

#### `add_pburst(colname=None, pause_colname=None, pause_method=None, pause_threshold=None, pause_a=None, iki_colname=None)`
Adds a column marking process bursts (pbursts) in the writing process. A pburst starts with a pause or the first character of a message (`'B'`), continues with uninterrupted typing (`'I'`), and is marked as `'O'` for events outside bursts or with missing pause information. Requires a pause column, which can be generated automatically if not provided.

#### `add_action(colname=None)`
Adds a column indicating the type of text change at each event: "addition", "deletion", "substitution", or "no change". This is determined by comparing the current and previous content of the message.

#### `add_span(colnames=[None, None, None, None], selection=[True,True,True,True])`
Adds columns marking the start and end positions of addition and deletion spans in the text, based on the detected action. The `selection` parameter allows you to choose which spans to compute.

#### `add_length(colnames=[None, None], selection=[True, True])`
Adds columns for the process length (number of events in a message) and product length (final text length) for each event. The `selection` parameter allows you to choose which lengths to compute.

### Burst and Pause Analysis

#### `burst_dataframe(burst_colname=None, iki_colname=None, action_colname=None)`
Returns a DataFrame with one row per burst, containing metrics such as burst duration, number of process characters, and product characters. Requires a burst column (e.g., from `add_pburst`).

#### `add_rburst()`
Adds columns for revision bursts (rbursts), which are segments of the writing process marked by revisions (e.g., deletions or substitutions). Implementation details depend on the revision detection logic.

#### `pburst_analysis(pburst_colname=None)`
Performs analysis on process bursts, returning metrics such as burst count, mean/median burst length, and distribution of burst types.

#### `pause_analysis(pause_colname=None, iki_colname=None)`
Analyzes pauses in the writing process, returning metrics such as pause count, mean/median pause duration, and pause rate per message or user.

### Additional Utilities

#### `_drop_anonymized()`
Removes events occurring in anonymized messages (for Language Hero data).

#### `_drop_nonsense()`
Removes events occurring in messages identified as nonsense (e.g., keysmashes or character repetitions).

#### `_drop_native()`
Removes events from native speakers (for Language Hero data).

---

## 1. Import the class and instantiate an empty instance of this class

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

## 2. Load and preprocess data 

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

## 3. Apply methods to analyse data

Besides the standard methods of pandas DataFrames, the class KeyLoggingDataFrames allows for specific methods related to writing process analysis to be performed on the data. The methods `add_iki()`, `add_pause()`, `add_action()`, `add_p_bursts()`, `add_r_bursts()` and `add_location()` add a column to the original dataframe, which can be analysed later. Make sure to specify column names that don't exist in the original column yet.

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

## Convert to 

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



## Technical Details: How Metrics Are Calculated

Below are the technical details for each method, including how metrics are computed and what each column represents. This section is intended for advanced users and developers who want to understand the implementation and logic behind each metric.

### `add_iki()`
- For each event, the IKI is calculated as the time difference (in milliseconds) between the current and previous key event within the same message. If `include_first_key` is `False`, the first event in each message is set to NaN. If `include_nontyping_events` is `False`, only events with `event_for_message_type == 0` are considered; otherwise, all events are included. The IKI column is filled with NaN for non-applicable events.

### `add_pause()`
- The pause column is a boolean indicating whether the IKI exceeds a threshold. For `method="fixed"`, the threshold is a constant (in ms). For `method="individualized"`, the threshold is computed as `a` times the user's median IKI. If IKI is NaN, pause is set to NaN. The method ensures that the pause column does not overwrite existing columns and checks for valid input.

### `add_pburst()`
- The pburst column marks process bursts: `'B'` for the start of a burst (pause is True or first character of a message), `'I'` for continuation (pause is False), and `'O'` for events outside bursts or with missing pause information. The method identifies the first character of each message and assigns burst labels accordingly.

### `add_action()`
- The action column is determined by comparing the current and previous content of the message. If the current content is longer, the action is "addition". If the content is unchanged, the action is "no change". Otherwise, the method checks for "deletion" or "substitution" by comparing character positions.

### `add_span()`
- The span columns mark the start and end positions of addition and deletion spans, based on the detected action. The method uses helper functions to find the first and last differing character between the current and previous content.

### `add_length()`
- The process length is the number of events in a message; the product length is the length of the final text. The method adds columns for these metrics, optionally controlled by the `selection` parameter.

### `burst_dataframe()`
- This method creates a DataFrame with one row per burst, including metrics such as burst duration (time between first and last event), process characters (number of events in the burst), and product characters (difference in text length between first and last event). Only events labeled as part of a burst are included.

### `add_rburst()`
- Revision bursts (rbursts) are segments of the writing process marked by revisions (e.g., deletions or substitutions). The method identifies rbursts based on the action column and groups consecutive revision events.

### `pburst_analysis()`
- Analyzes process bursts, returning metrics such as the number of bursts, mean and median burst length, and the distribution of burst types. The analysis is based on the pburst column.

### `pause_analysis()`
- Computes pause metrics, including the number of pauses, mean and median pause duration, and pause rate per message or user. The analysis uses the pause and IKI columns.

---


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



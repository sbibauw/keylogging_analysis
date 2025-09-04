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

## analysis



# Documentation `keylogging_analysis version 0.0.1`




## preprocessing

> ### load_files(events, messages, conversations)
> Load the csv files containing the events, messages and conversations and return them as a single merged file
> input: `events`, `messages`, `conversations` contain the relative or absolute path to the respective.csv files
> output: merged dataframe of class KeyLoggingDataFrame (which supports the same functions as `pandas` dataframes, but with additional functions
> warnings: if id's are missing, missing columns), the function only merges on the available subset and issues a warning specifying that the data may be incomplete
> warnings: if the original dataframes contain double items

> ### kldf.select(users, conversations)???
> Returns a subset of users or conversations
> input
> ??? method to select only native speakers

> ### kldf.remove_nonsense()
> Removes nonsense messages from the dataset
> !! adapt model for LL dataset

> ### kldf.remove_nontyping()

## analysis

> ### kldf.add_ikis(name="iki")
> Add a column "iki" to the merged dataframe, containing for each event the Inter-Keystroke Interval, or the duration between the current and the previous keystroke, expressed in milliseconds.
> Warning: the function issues a warning if the column "iki" exists already

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

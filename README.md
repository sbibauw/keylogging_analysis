# About `keylogging_analysis`

`keylogging_analysis` is a Python module aimed at processsing and analysing data from the **Language Hero** ("lh") and **Language Lab** ("ll") datasets. It is useful to researchers interested in **L2 language fluency**, **chatbot-student interactions** or **keylogging data**.

# How to install?

(pip)

# Language Hero and Language Lab data


# Documentation keylogging_analysis version 1.0

The functions proposed in this module can be distinguished in three categories
1. preprocessing
2. analysis
3. convertion

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

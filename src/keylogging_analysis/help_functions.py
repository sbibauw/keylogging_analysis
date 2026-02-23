import pandas as pd 

def generate_colname(base, existing_cols):
    """
    input: base name (str), list of existing column names (list of str)
    output: a new column name that is not in existing_cols, by appending _1, _2, etc. if necessary
    """
    if base not in existing_cols:
        return base
    else:
        i = 1
        new_name = f"{base}{i}"
        while new_name in existing_cols:
            i += 1
            new_name = f"{base}{i}"
        return new_name



def detect_action(curr, prev):
    """input: two strings, the current and previous content of a message
    output: 0 = "addition", 1 = "deletion", 2 = "substitution" or 3 = "no change"
    Handle NA/None values by converting them to empty strings.
    """
    # Handle NA/None values
    curr = "" if pd.isna(curr) else str(curr)
    prev = "" if pd.isna(prev) else str(prev)
    
    if len(curr) > len(prev):
        return "addition"
    if curr == prev:
        return "no change"
    else:  # differ between deletion and substitution
        # Find first position in which curr and prev differ: i
        i = 0
        while i < min(len(curr), len(prev)) and curr[i] == prev[i]:
            i += 1

        # how many characters are there in curr after the first different character
        curr_after = len(curr) - i

        if curr[i:] == prev[len(prev)-curr_after:]:
            return "deletion"
        else:
            return "substitution"
        
def detect_start_deletion_span(curr, prev):
    """input: two strings, the current and previous content of a message
    output: 
            for deletion: position of first deleted character in prev
            for addition: None
            for substitution: position of substituted character in prev
            for no change: None"""
    curr = "" if pd.isna(curr) else str(curr)
    prev = "" if pd.isna(prev) else str(prev)
    if curr == prev:
        return None
    action = detect_action(curr, prev)
    if action in ["no change", "addition", None]:
        return None
    i = 0
    while i < min(len(curr), len(prev)) and curr[i] == prev[i]:
            i += 1
    return i

def detect_start_addition_span(curr, prev):
    """input: two strings, the current and previous content of a message
    output: 
            for addition: position of first added character in curr
            for deletion: position of first deleted character in prev
            for substitution: position of substituted character in curr
            for no change: -1"""
    curr = "" if pd.isna(curr) else str(curr)
    prev = "" if pd.isna(prev) else str(prev)
    if curr == prev:
        return None
    action = detect_action(curr, prev)
    if action in ["no change", "deletion", None]:
        return None
    i = 0
    while i < min(len(curr), len(prev)) and curr[i] == prev[i]:
            i += 1
    return i

def detect_end_addition_span(curr, prev):
    """input: two strings, the current and previous content of a message
    output: 
            for addition: position of last added character in curr
            for deletion: position of last deleted character in prev
            for substitution: position of last substituted character in curr
            for no change: -1"""
    curr = "" if pd.isna(curr) else str(curr)
    prev = "" if pd.isna(prev) else str(prev)
    if curr == prev:
        return None
    action = detect_action(curr, prev)
    if action in ["no change", "deletion", None]:
        return None
    #j1 = len(curr)
    #j2 = len(prev)
    #while j1 > 0 and j2 > 0 and curr[j1-1] == prev[j2-1]:
    #    j1 -= 1
    #    j2 -= 1
    else: 
        # only works if only one character added (which is the case in the datasets we have)
        return detect_start_addition_span(curr, prev) + 1

def detect_end_deletion_span(curr, prev):
    """input: two strings, the current and previous content of a message
    output: 
            for deletion: position of last deleted character in prev
            for addition: None
            for substitution: position of last substituted character in prev
            for no change: None"""
    curr = "" if pd.isna(curr) else str(curr)
    prev = "" if pd.isna(prev) else str(prev)
    if curr == prev:
        return None
    action = detect_action(curr, prev)
    if action in ["no change", "addition", None]:
        return None
    j1 = len(curr)
    j2 = len(prev)
    while j1 > 0 and j2 > 0 and curr[j1-1] == prev[j2-1]:
        j1 -= 1
        j2 -= 1
    return j2

def get_burst_metrics(events, burst_colname:str=None, action_colname:str=None, iki_colname:str=None):
    """ returns a dataframe with one row per burst, containing various metrics (duration, process_characters, product_characters)
    input: events is a KeyLogggingDataFrame containing events of one message only"""
    # verify parameters
    if not burst_colname:
        raise ValueError("burst_colname must be specified.")
    if burst_colname not in events.columns:
        raise ValueError(f"Burst column '{burst_colname}' not found in DataFrame.")
    if not action_colname:
        raise ValueError("action_colname must be specified.")
    if action_colname not in events.columns:
        raise ValueError(f"Action column '{action_colname}' not found in DataFrame.")
    if not iki_colname:
        raise ValueError("iki_colname must be specified.")
    if iki_colname not in events.columns:
        raise ValueError(f"IKI column '{iki_colname}' not found in DataFrame.")
    if not all(events[burst_colname].isin(['B', 'I', 'O', None])):
        raise ValueError(f"Burst column '{burst_colname}' must contain only 'B', 'I', 'O' or None values.")
    if len(events) == 0:
        return pd.DataFrame()  # return empty dataframe if no events
    if len(events['message_id'].unique()) != 1:
        raise ValueError("Events must belong to the same message (i.e. have the same message_id).")
    
    # create empty dataframe for burst metrics
    df = pd.DataFrame(columns=['message_id', 'burst_order', 'process_chars', 'product_chars'])
    
    # select only B and I events (O-events are outside a burst)))
    events_message = events[events[burst_colname].isin(['I', 'B'])]
    burst_order = 0
    for j in range(len(events_message)):
        if events_message.iloc[j][burst_colname] == "B":
            burst_order +=1
            # select all events of burst
            b_burst = pd.DataFrame(events_message.iloc[j]).T
            i_burst = events_message.iloc[j+1:].loc[   # all I-events after B ! additions and substitutions only !
                lambda x: (x[burst_colname] == 'I').cumprod().astype(bool)
            ]
            events_burst = pd.concat([b_burst, i_burst], axis=0)
            
            # get burst metrics
            message_id = events_message.iloc[j]["message_id"]
            process_chars = len(events_burst)
            # get product_cars: difference in length between last and first event of burst
            product_chars = len(events_burst.iloc[-1]["content"]) - len(events_burst.iloc[0]["content"])
            

            row = pd.Series({
                'message_id': events_message.iloc[j]["message_id"],
                'burst_order': burst_order,
                'process_chars': process_chars,
                'product_chars': product_chars
            })
            df = pd.concat([df, row.to_frame().T], ignore_index=False)
    return df

def detect_distance_to_end(content, end_deletion, end_addition):
    """input: content (str), end_deletion (int or None), end_addition (int or None)
    output: distance to end of content, defined as the number of characters after the last change (deletion or addition)
    if both end_deletion and end_addition are None, return None
    if content is None or NaN, return None
    """
    if not content:
        content = ""
    content = str(content)
    if end_deletion is None and end_addition is None:
        return None
    if end_deletion is not None and end_addition is not None:
        last_change = max(end_deletion, end_addition)
    elif end_addition is None:
        last_change = end_deletion
    else:
        last_change = end_addition
    return len(content) - last_change


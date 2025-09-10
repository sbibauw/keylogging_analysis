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


def are_ids_unique(df):
    """
    input: polars dataframe containing a column "id"
    output: True if all ids are unique, False if not
    """
    if len(df) == df.n_unique(subset="id"):
        return True
    else:
        ids = df.group_by("id").agg(pl.count()).filter(pl.col("count") >= 2)
        items = []
        for id in ids["id"]:
            items_id = df.filter(pl.col("id") == id)
            items.append(items_id)
        items = pl.concat(items)
        items.write_csv("inspect_non_zero_ids.csv")
        # print(items)
        # with pl.Config(tbl_rows=-1):
        #     print(items.group_by("SCENARIO_ID").agg(pl.count()).sort(by="count", descending=True))
        # with pl.Config(tbl_rows=-1):
        #     print(items.group_by("TASK_ID").agg(pl.count()).sort(by="count", descending=True))
        return False

    
def all_of_first_in_last(col1, col2):
    """
    input: two lists
    output: returns True if all elements of col1 appear at least once in col2
    e.g. to verify whether ids are valid (see use in main.py)
    """
    col1 = col1.unique()
    col2 = col2.unique()
    return col1.is_in(col2).all()

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
        
def detect_start_deletion_span(curr, prev, action):
    """input: two strings, the current and previous content of a message
    output: 
            for deletion: position of first deleted character in prev
            for addition: None
            for substitution: position of substituted character in prev
            for no change: None"""
    if not curr:
        curr = ""
    if not prev:
        prev = ""
    if curr == prev:
        return None
    if action in ["no change", "addition", None]:
        return None
    i = 0
    while i < min(len(curr), len(prev)) and curr[i] == prev[i]:
            i += 1
    return i

def detect_start_addition_span(curr, prev, action):
    """input: two strings, the current and previous content of a message
    output: 
            for addition: position of first added character in curr
            for deletion: position of first deleted character in prev
            for substitution: position of substituted character in curr
            for no change: -1"""
    if not curr:
        curr = ""
    if not prev:
        prev = ""
    if curr == prev:
        return None
    if action in ["no change", "deletion", None]:
        return None
    i = 0
    while i < min(len(curr), len(prev)) and curr[i] == prev[i]:
            i += 1
    return i


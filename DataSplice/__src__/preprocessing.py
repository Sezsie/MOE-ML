#############################################################################################################
# DOCUMENTATION
#############################################################################################################
# AUTHOR: Sabrina
# LAST UPDATED: 2024-08-04
#
# FUNCTION: This script is designed to perform some final preprocessing steps on the dataset and produce a new, more refined dataset.
#
# INPUT: Our original uncleaned combined dataset, containing both command and conversational features.
# OUTPUT: A new dataset that has been cleaned of punctuation, stopwords, and other irrelevant features.
#############################################################################################################

# imports
import os
import pandas as pd # we can use this to load csv files, handy!
# import nltk # we would have used this for stopwords, but we don't need it anymore since our vectorizer does it for us
import re # regex for removing unwanted characters

# define the desktop path using os
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

# we can just assume that the combined dataset is on the desktop
combined_dataset_path = os.path.join(desktop_path, 'combined_dataset.csv')

# load into a dataframe
df = pd.read_csv(combined_dataset_path)

# extract the text column
text_column = df['text']

# grab the number of rows in the dataset
num_rows = df.shape[0]

# how many rows have been processed
rows_processed = 0

# self explanatory function name is self explanatory
def remove_punctuation(text):
    # remove any punctuation except for periods, we will have special logic for those
    text = re.sub(r'[^\w\s.]', '', text)
    
    # if a period is followed by a space, remove the period
    text = re.sub(r'\. ', ' ', text)
    
    return text

# combines both functions to preprocess the text
def preprocess_text(text):
    global rows_processed
    global num_rows
    print(f"Preprocessing text {rows_processed}/{num_rows}...")
    # increment the rows processed
    rows_processed += 1
    
    # apply both functions to the text
    return remove_punctuation(text)

# apply the preprocessing function to the text column
df['text'] = text_column.apply(preprocess_text)

# save the new dataset to the desktop
new_dataset_path = os.path.join(desktop_path, 'cleaned_dataset.csv')

# save the new dataset to the desktop
df.to_csv(new_dataset_path, index=False)

print(f"The cleaned dataset has been saved to {new_dataset_path}!")




 
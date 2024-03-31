
#############################################################################################################
# DOCUMENTATION
#############################################################################################################
# AUTHOR: Garrett Thrower
# LAST UPDATED: 2024-30-3
# FUNCTION: This script is designed to splice two existing datasets and create a new dataset for the purpose of training a text intent classification model.
# INPUT: Two datasets, one containing a command dataset and the other containing a conversation dataset.
# OUTPUT: A new dataset containing a combination of the two datasets, with the command dataset being the primary dataset. Classes are "command" and "conversational". (0 and 1 respectively)
# 
# PURPOSE: To create a dataset for the purpose of training a text classification model.
#############################################################################################################

# imports

from contact_openai import AIHandler, Agent
from utils import Utilities, DebuggingUtilities
from sklearn.preprocessing import MinMaxScaler

import re
import tabulate
import os
import pandas as pd # we can use this to load csv files, handy!
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console

# create a new AIHandler object
utilities = Utilities()
debug = DebuggingUtilities()
console = Console()
dprint = debug.dprint

# define the desktop path using os
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

# define the path to the command dataset
command_dataset_path = os.path.join(desktop_path, 'commands.csv') 

# now for the conversation dataset
conversation_dataset_path = os.path.join(desktop_path, 'conversations.csv')

# load both datasets into pandas dataframes
command_df = pd.read_csv(command_dataset_path)
conversation_df = pd.read_csv(conversation_dataset_path)


# we want to extract the command text from the command dataset and the conversation text from the conversation dataset
# for each of these datasets respectively, the text that we want is under the 'Text' column in the command dataset and the 'question' column in the conversation dataset
command_text = command_df['Text']
conversation_text = conversation_df['question']

# now we want to combine these two datasets into a new dataset. We will use the command dataset as the primary dataset and the conversation dataset as the secondary dataset
# our new dataset will have the following columns: 'text' and 'label'.
# since we know the purposes of each dataset, we can label the command dataset as 'command' and the conversation dataset as 'conversation'
# 0 = command, 1 = conversational     

# create a new dataframe with the text and label columns
new_df = pd.DataFrame(columns=['text', 'label'])

# use tabulate to display the percentage of labels in the dataset
def display_distribution(dataframe):
    # get the value counts of the 'label' column
    value_counts = dataframe['label'].value_counts()

    # get the percentage of each label
    percentage = value_counts / value_counts.sum() * 100

    # create a table using tabulate
    table = tabulate.tabulate(zip(value_counts.index, value_counts, percentage), headers=['Label', 'Count', 'Percentage'], tablefmt='fancy_grid')
    
    # 

    # print the table
    print(table)
    

def main():

    # create dataframes from the command and conversation text
    command_df = pd.DataFrame({'text': command_text, 'label': 0})
    conversation_df = pd.DataFrame({'text': conversation_text, 'label': 1})

    # reset the indices of the dataframes
    command_df.reset_index(drop=True, inplace=True)
    conversation_df.reset_index(drop=True, inplace=True)

    # interleave the dataframes
    new_df = pd.concat([command_df, conversation_df]).sort_index(kind='merge')

    # lower the text in the 'text' column
    new_df['text'] = new_df['text'].apply(lambda x: x.lower())
    
    # now that we are done, we can save the new dataframe to a csv file
    new_df.to_csv('combined_dataset.csv', index=False)
    
    display_distribution(new_df)

main()  # run the main function
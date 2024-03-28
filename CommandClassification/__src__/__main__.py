
#############################################################################################################
# DOCUMENTATION
#############################################################################################################
# AUTHOR: Garrett Thrower, Colby McClure, Brian Boggs
# LAST UPDATED: 2024-27-03
# FUNCTION: This script is designed to generate a synthetic dataset for the purpose of training a classifier.
# It will take a pre-existing small dataset and generate a larger dataset by using a GPT-3.5 model to generate it.
#
# PURPOSE: To generate large amounts of synthetic data for the purpose of training a classifier.
#############################################################################################################
# imports

from contact_openai import AIHandler, Agent
from utils import Utilities, DebuggingUtilities

# create a new AIHandler object
utilities = Utilities()
debug = DebuggingUtilities()
ai = AIHandler()
dprint = debug.dprint


# prompt for the agent
prompt = """
Write one sentence of text in natural language that are either commands/requests or conversational in nature.
Please follow the below rules without exception:
    - Ensure that outputs are varied in structure and content.
    - Variation in the output should be formatted as a request or command.
    - Your output should be similar in theme to the input, but completely different in structure and content.
    - The output should be from the perspective of a user requesting that a computer performs a task.
    - Do not include any commands that could not be realistically be accomplished by a computer.
    - Do not include any commands that would require an internet connection.
"""

agentname = "DataGen"
agentmodel = "gpt-3.5-turbo-instruct"

# agent settings

temperature = 1 # this setting controls the randomness of the output. 0.0 is deterministic, 1.0 is maximum randomness.
max_tokens = 1000 # this setting controls the maximum number of tokens that the model can output.
frequency_penalty = 1 # this setting controls the diversity of the output.
presence_penalty = 1 # this setting controls the diversity of the output.

# create the agent
dprint("Creating agent...")
DataGen = ai.createAgent(
    agentname=agentname,
    agentmodel=agentmodel,
    systemprompt=prompt,
    temperature=temperature,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
    max_tokens=max_tokens
)

command_table = ["Open my browser."]
conversation_table = ["How are you doing today?"]

# a function that takes two tables of strings, passes each string in each table to the GPT-3.5 model, and returns the output
# as two lists of strings. This function is used to generate the data for the synthetic dataset.
# makes use of the AI's chat function to generate the data.
# sample_size is the number of samples to generate for each table.
def generateData(input_table, input_table2, sample_size=1):
    generated_commands = []
    generated_conversationals = []
    for command in input_table:
        dprint(f"Generating command: {command}")
        for i in range(sample_size):
            response = DataGen.chat(command)
            # clean response of newline characters
            response = response.replace("\n", "")
            generated_commands.append(response)

    return generated_commands, generated_conversationals
        

samplesPerEntry = 3 # for every entry in the command_table and conversation_table, generate this many extra samples


def main():
    print("Generating data...")
    # generate data
    generated_commands, generated_conversationals = generateData(command_table, conversation_table, samplesPerEntry)
    print("Generated Commands: ", generated_commands)
    print("Generated Conversationals: ", generated_conversationals)

    
main()  # run the main function
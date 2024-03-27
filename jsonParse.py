import json 

# Change this to whatever file you need to parse
filename = 'data.json'

# Load the data into the 'data' variable
with open(filename, 'r') as file: 
    data = json.load(file)


# Declare our lists and our keys to search for
conversationalList = []
commandList = []

commandKey = 'command'
conversationKey = 'conversational' 

# For each key in the data, check if it's a command or conversational key
# If it is, append it to the appropriate list
for key in data:
    if key == commandKey:
        commandList.append(data[key])
    elif key == conversationKey:
        conversationalList.append(data[key])

# Print the lists
print('Command List: ', commandList)
print('Conversational List: ', conversationalList)

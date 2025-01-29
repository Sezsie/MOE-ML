
#############################################################################################################
# DOCUMENTATION
#############################################################################################################
# AUTHOR: Sabrina
# LAST UPDATED: 2024-30-3
#
# FUNCTION: A fun little script that tests the model we trained in vectorizer.py. User is prompted to enter a sentence, and the model will predict whether it is a command or a conversational sentence.
#############################################################################################################

import os
import pickle


# point to the desktop
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

# load the model
model_path = os.path.join(desktop_path, 'CommVConvModel.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
# load the vectorizer
vectorizer_path = os.path.join(desktop_path, 'CommVConvVectorizer.pkl')
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)
    
# prompt the user to enter a sentence
while True:
    sentence = input("Enter a sentence: ")
    
    # vectorize the sentence
    vectorized_sentence = vectorizer.transform([sentence])
    
    # predict the sentence
    prediction = model.predict(vectorized_sentence.toarray())
    
    # print the prediction
    if prediction == 0:
        print("Command")
    else:
        print("Conversational")
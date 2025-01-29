#############################################################################################################
# DOCUMENTATION
#############################################################################################################
# AUTHOR: Sabrina
# LAST UPDATED: 2024-08-04
#
# FUNCTION: This script reads a spliced dataset and splits it into training (70%) and testing (30%) sets,
# vectorizes the features w/ 'bag of n-grams', then trains a Multionmial Naive Bayes so we can classify the accuracy of the approach.
#
# INPUT: One spliced dataset, containing both command and conversational features.
# OUTPUT: The accuracy of a Naive Bayes model trained using 'bag of n-grams' feature reduction.
#############################################################################################################

# general imports
import os

# strucural imports
import numpy as np
import pandas as pd
from tabulate import tabulate

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# dimensionality reduction imports
from sklearn.decomposition import PCA

# balancing imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

# pickling imports
import pickle

# visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# analysis imports 
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# settings
seed = 42 # random.randint(0, 1000)
desired_samples = 10000 # desired number of samples for each class
conversation_sample_boost = 1500 # how many extra samples to add to the conversational class when oversampling. we are doing this because even when the classes are balanced, the model still has a bias towards the command class.

# define path to desktop and path to dataset. currently using example dataset.
desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop') 
dataset_path = os.path.join(desktop_path, 'cleaned_dataset.csv')

# saves the model and vectorizer to the desktop
def save_model(model, vectorizer, desktop_path):
    # save the bayes model to the desktop using pickle
    print(f"Saving {model} model to desktop...")
    model_path = os.path.join(desktop_path, 'CommVConvModel.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print("Model saved to desktop.")

    # save the vectorizer to the desktop using pickle
    print("Saving vectorizer...")
    vectorizer_path = os.path.join(desktop_path, 'CommVConvVectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Vectorizer saved to desktop.")

# load the dataset into a dataframe, then split the data and target feature into seperate columns
df = pd.read_csv(dataset_path)
X = df['text']
y = df['label']

print(f"Seed: {seed}")

# split data and target feature for training and testing with a 80/20 split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

#############################################################################################################
# FEATURE EXTRACTION
#############################################################################################################

# create a bag of words with unigrams and bigrams
tf = TfidfVectorizer(analyzer = 'word', stop_words='english', max_features=1000)

# convert training data to bag of words
X_train_cv = tf.fit_transform(X_train)
X_test_cv = tf.transform(X_test)

# print the original shape of the data
print(f"Original shape of data: {X_train_cv.shape}")

# Perform PCA on the dataset before balancing
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_cv.toarray())

# Plot the first two PC's of the dataset in a scatter plot
for i, target_name in enumerate(y_train.unique()):
    plt.scatter(X_pca[y_train == target_name, 0], X_pca[y_train == target_name, 1], label=target_name)

plt.legend()
plt.title('PCA of the dataset (Pre-balancing)')
plt.grid(True)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

#############################################################################################################
# BALANCING THE DATASET
#############################################################################################################

# oversample minority class (1 = conversational) using SMOTE
print(f"Class being oversampled: {y_train.value_counts().idxmin()}")
print(f"Oversampling...")
sm = SMOTE(sampling_strategy={y_train.value_counts().idxmin(): desired_samples + conversation_sample_boost}, random_state=seed)
X_train_cv, y_train = sm.fit_resample(X_train_cv, y_train)

# undersample majority class (0 = command) using ClusterCentroids
print(f"Undersampling...")
cc = ClusterCentroids(sampling_strategy={y_train.value_counts().idxmax(): desired_samples}, random_state=seed)
X_train_cv, y_train = cc.fit_resample(X_train_cv, y_train)

# Perform PCA on the dataset after balancing the classes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_cv.toarray())

# Plot the first two PC's of the dataset in a scatter plot
for i, target_name in enumerate(y_train.unique()):
    plt.scatter(X_pca[y_train == target_name, 0], X_pca[y_train == target_name, 1], label=target_name)

plt.title('PCA of the dataset (Post-balancing)')
plt.legend()
plt.grid(True)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

#############################################################################################################
# TRAINING & EVALUATION
#############################################################################################################

models = []
models.append(('NB', MultinomialNB()))
models.append(('LR', LogisticRegression(solver = 'liblinear', multi_class = 'ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 2, weights = 'distance', metric = 'hamming', algorithm = 'brute')))
models.append(('RF', RandomForestClassifier(n_estimators = 100, max_depth = 10, random_state = seed)))

results = []
names = []
score_table = []

desired_model = None # the model that will be saved to the desktop

for name, model in models:
    print(f"Training {name}...")
    kfold = StratifiedKFold(n_splits = 10, random_state = seed, shuffle = True)
    cv_results = cross_val_score(model, X_train_cv.toarray(), y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    
    # train the model and add the trained model to the models list
    model.fit(X_train_cv.toarray(), y_train)
    
    # make predictions
    y_pred = model.predict(X_test_cv.toarray())

    # Create the confusion matrix based off model predictions
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure() 
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} Confusion Matrix')
    plt.show() 
    
    # calculate precision, recall, and F1 score
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    #  add the scores to the score table
    score_table.append([name, precision, recall, f1])
    
    # set desired_model to the model with a criteria of my choice, in this case just referencing by name
    if name == 'NB':
        desired_model = model
        save_model(desired_model, tf, desktop_path)

# plot boxplot for algorithm comparison
plt.boxplot(results, labels = names)
plt.title('Algorithm Comparison')
plt.show()

# create a table with precision, recall, and F1 score for each model
table = tabulate(score_table, headers=['Model', 'Precision', 'Recall', 'F1 Score'], tablefmt='fancy_grid')

# print the table
print(table)

print("Done.")




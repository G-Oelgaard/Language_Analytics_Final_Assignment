## Importing packages ##
# data processing tools
import os, sys, re
sys.path.append(os.path.join(".."))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import urllib.request
import pandas as pd
from tqdm import tqdm
import utils.classifier_utils as clf
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

from scipy.special import softmax

from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
from transformers import pipeline, set_seed

#args_parse
import argparse

## Functions ##
# Load data
def data_load():
    filename = os.path.join("..", "in", "last_statement_texas.csv")
    data = pd.read_csv(filename, sep=';')
    data.dropna(subset=['Last statement'], inplace=True) # removes all prisoners that did not give any statements
    data.reset_index(drop=True, inplace=True)
    
    return data
    
# format labels
def format_label(label, data):
    if label == "Race":
        data.rename(columns = {'Race':'label'}, inplace = True)
        others = data[(data.label == 'Other')].index
        data.drop(labels = others, axis=0, inplace=True)
        data.reset_index(drop=True, inplace=True)
        label_names = ["Black","Hispanic","White"]
        return label_names, data
    else:
        age_group = []
        for idx, row in tqdm(data.iterrows()):
            Age = row["Age"]
            int(Age)
            if Age >= 40:
                group = "40 or over"
                age_group.append(group)
            else:
                group = "39 or under"
                age_group.append(group)
        data["label"] = age_group
        label_names = ["39 or under","40 or over"]
        return label_names, data

# format sentiment
def sentiment_format(data):
    classifier = pipeline("text-classification", 
                  model="j-hartmann/emotion-english-distilroberta-base", 
                  return_all_scores=True)
    data["Last statement"].astype(str)

    all_scores = []
    print("Getting sentiment scores:")
    for idx, row in tqdm(data.iterrows()):
        last_statement = row["Last statement"][:512] # model has string limit of 512
        predictions = classifier(last_statement)
        all_scores.append(predictions)
    data["sentiment"] = all_scores

    inmate_scores = []
    for index, lists in enumerate(all_scores):
        lists = lists[0]  
        inmate_scores.insert(index, [dict["score"] for dict in lists])
    inmate_scores = np.array(inmate_scores)
    return inmate_scores, data

# balance data
def data_balance(label, data):
    if label == "Race":
        data = clf.balance(data, 91)
        return data
    else:
        data = clf.balance(data, 203)
        return data

# Sentiment - Logistical Regression
def sentiment_log(inmate_scores, data,label, label_names, save_model):
    X = inmate_scores
    y = data["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y,
                                                    test_size= 0.2,
                                                    random_state=42)
    
    classifier = LogisticRegression(random_state = 42,).fit(X_train,y_train)
    
    y_pred = classifier.predict(X_test)
    
    outpath = os.path.join("..", "out", label+"_sentiment_LogReg_") # save outpath
    
    classifier_metrics = metrics.classification_report(y_test,y_pred, target_names = label_names) # create and print class report
    print(classifier_metrics) 
    
    with open(outpath+"classification_report.txt","w") as file: # save class report
        file.write(str(classifier_metrics))
    
    title = "Learning Curves (Sentiment - Logistical regression)" # create learning curves
    cv = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)
    
    estimator = LogisticRegression(random_state=42)
    
    plt.figure()
    clf.plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=16) # plot learning curves
    plt.savefig(outpath+"learning_curves.jpg") # save plot
    
    if save_model == None:
        pass
    else:
        dump(classifier, "../out/"+save_model+".joblib")

# Sentiment - Neural Network
def sentiment_NN(inmate_scores, data, label, label_names, save_model):
    X = inmate_scores
    y = data["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size= 0.2,
                                                    random_state=42)
    
    if label == "Race":
        mlp = MLPClassifier(random_state = 42, # Set classifier
                        alpha = 0.01,
                        hidden_layer_sizes = (32,),
                        activation = "relu",
                        max_iter = 3000
                       )
    else:
        mlp = MLPClassifier(random_state = 42, # Set classifier
                        alpha = 0.01,
                        hidden_layer_sizes = (32,),
                        activation = "logistic",
                        max_iter = 3000
                       )
    
    mlp.fit(X_train, y_train)
    
    y_pred = mlp.predict(X_test)
    
    outpath = os.path.join("..", "out", label+"_sentiment_NN_")
    
    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = label_names) # create and print class report
    print(classifier_metrics)
    
    with open(outpath+"classification_report.txt","w") as file: # save class report
        file.write(str(classifier_metrics))
    
    title = "Learning Curves (Sentiment Neural Network - MLP)" # create learning curves
    cv = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)
    
    plt.figure()
    clf.plot_learning_curve(mlp, title, X, y, cv=cv, n_jobs=16) # plot learning curves
    plt.savefig(outpath+"learning_curves.jpg") # save plot
    
    if save_model == None:
        pass
    else:
        dump(mlp, "../out/"+save_model+".joblib")

# Statement - Logistical Regression
def class_log(data, label, label_names, save_model):
    X = data["Last statement"]
    y = data["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, # split
                                                    y,
                                                    test_size= 0.2,
                                                    random_state=42)
    
    vectorizer = TfidfVectorizer(ngram_range=(1,10), # set vectorizer
                             lowercase=True,
                             max_df=0.99,
                             min_df=0.01,
                             max_features=1000)
    
    X_train_feats = vectorizer.fit_transform(X_train) # fit vectorizer
    X_test_feats = vectorizer.transform(X_test)
    
    classifier = LogisticRegression(random_state = 42).fit(X_train_feats,y_train) # set classifier (logReg)
    
    y_pred = classifier.predict(X_test_feats)
    
    outpath = os.path.join("..","out", label+"_statement_LogReg_")
    
    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = label_names) # create and print class report
    print(classifier_metrics)
    
    with open(outpath+"classification_report.txt","w") as file: # save class report
        file.write(str(classifier_metrics))
        
    X_vect = vectorizer.fit_transform(X)
    
    title = "Learning Curves (Statement - Logistical regression)" # create learning curves
    cv = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)

    estimator = LogisticRegression(random_state=42)
    
    plt.figure()
    clf.plot_learning_curve(estimator, title, X_vect, y, cv=cv, n_jobs=16) # plot learning curves
    plt.savefig(outpath+"learning_curves.jpg") # save plot
    
    if save_model == None:
        pass
    else:
        dump(classifier, "../out/"+save_model+".joblib")

# Statement - Neural Network
def neural_net(data, label, label_names, save_model):
    X = data["Last statement"]
    y = data["label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, # split
                                                    y,
                                                    test_size= 0.2,
                                                    random_state=42)
    
    vectorizer = TfidfVectorizer(ngram_range=(1,10), # set vectorizer
                             lowercase=True,
                             max_df=0.99,
                             min_df=0.01,
                             max_features=1000)
    
    X_train_feats = vectorizer.fit_transform(X_train) # fit vectorizer
    X_test_feats = vectorizer.transform(X_test)
    
    if label == "Race":
        mlp = MLPClassifier(random_state = 42, # Set classifier
                        alpha = 0.01,
                        hidden_layer_sizes = (32,),
                        activation = "relu",
                        max_iter = 3000
                       )
    else:
        mlp = MLPClassifier(random_state = 42, # Set classifier
                        alpha = 0.01,
                        hidden_layer_sizes = (32,),
                        activation = "logistic",
                        max_iter = 3000
                       )

    mlp.fit(X_train_feats, y_train)
    
    y_pred = mlp.predict(X_test_feats)
    
    outpath = os.path.join("..","out", label+"_statement_NN_")
    
    classifier_metrics = metrics.classification_report(y_test, y_pred, target_names = label_names)
    print(classifier_metrics)
    
    with open(outpath+"classification_report.txt","w") as file: # save class report
        file.write(str(classifier_metrics))
        
    X_vect = vectorizer.fit_transform(X)
    
    title = "Learning Curves (Statement Neural Network - MLP)" # create learning curves
    cv = ShuffleSplit(n_splits=100,test_size=0.2,random_state=0)

    plt.figure()
    clf.plot_learning_curve(mlp, title, X_vect, y, cv=cv, n_jobs=16) # plot learning curves
    plt.savefig(outpath+"learning_curves.jpg") # save plot
    
    if save_model == None:
        pass
    else:
        dump(mlp, "../out/"+save_model+".joblib")
    
# args_parse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--label", choices=['Race', 'Age'], required = True, help="What label do you want to use? 'Race' or 'Age'?")
    ap.add_argument("-i", "--input_data", choices=['Statement', 'Sentiment'], required = True, help="What do you want to use for the classification? 'Statement' or 'Sentiment'?")
    ap.add_argument("-c", "--class_type", choices=['LogReg', 'NN'], required = True, help="What type of classification do you want to use? Logistical regression ('LogReg') or a simple neural network ('NN')?")
    ap.add_argument("-s", "--save_model", required = False, help="What you want the model to be saved as. Do not give af file extension! If no argument is given, the model won't save")
    args = vars(ap.parse_args())
    return args

## Main ##
# Defining main
def main():
    args = parse_args()
    data = data_load()
    label_names, data = format_label(args["label"], data)
    if args["input_data"] == "Sentiment":
        inmate_scores, data = sentiment_format(data)
    else:
        data = data_balance(args["label"], data)
    if args["input_data"] == "Sentiment":
        if args["class_type"] == "LogReg":
            sentiment_log(inmate_scores, data, args["label"], label_names, args["save_model"])
        else:
            sentiment_NN(inmate_scores, data, args["label"], label_names, args["save_model"])
    else:
        if args["class_type"] == "LogReg":
            class_log(data, args["label"], label_names, args["save_model"])
        else:
            neural_net(data, args["label"], label_names, args["save_model"])

# Running main
if __name__ == "__main__":
    main()
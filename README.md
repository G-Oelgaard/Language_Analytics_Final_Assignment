# Language_Analytics_Final_Assignment
## ------ SCRIPT DESCRIPTION ------
This repository contains a script that trains a LogReg or simple NN model to predict the race or age of inmates based on either the text or sentiment of their final statement. 

The script will:
- Load the relevant data.
    - If the user has chosen to use the text statements, the script will also balance the data.
    - If the user has chosen to use sentiment, the model will use the huggingface model "emotion-english-distilroberta-base" to predict one of 7 different emotions.
- Train either a logReg og NN model:
    - The user can specify if the model should try to predict age ('40 or over' / '39' or under) or race ('Black', 'Hispanic' or 'White') 
- Print a classification report to show how good the classifier is.
- Save a learning curve plot to visualize the cross-validation of the model.
- If specified the script can also save the model for later use.

## ------ METHOD ------
While this script uses both logistical regression and simple neural networks, it was originally intended to use CNN. However, after running some preliminary tests, it rapidly became apparent that it was both a slower and score-wise worse approach. It was therefore changed to its current format with the user designating what model to use. In whole, this model was created with the intention of giving the user a wide possibility of choices. Almost all, except the variables in the vectorizer, can be defined when starting the model. The reasoning for not being able to change the vectorizer, is because the variables are already set to the parameters that gave the best scores in training. 

A long time was also devoted to trying to figure out a way to balance the dataset when using sentiment scores. But due to built-in limitations with the clf.balance function, no way was found. It most likely could have been possible with a for-loop counting both index and labels until a certain requirement was met. But I could not see a way where this would not have led to the creation of the exact same balanced dataset each time. In other words: i could not see how this could be done without permanently excluding a large part of the dataset.

## ------ DATA ------
The data is a list of last statements made by deathrow inmates in Texas from 1982-2022. 

The data can be found here: https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html

However, as the data cannot be downloaded from the website, a webscraping script is needed. This can be done by the user or by using the R-markdown script "Webscraping_statements.rmd" created by myself.

If webscraped by the user, the following in the column "Last Statement" needs to be transformed to "NA": "No statement was made", "None", "Spoken: No", "No statement given", "This inmate declined to make a last statement", "No, i have no final statement", "No" and "No last statement".

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py script and .rmd webscraping script

"in" FOLDER:
- This is where the data used in the script should be placed. Ie. where the the "last_statement_texas.csv" file should be placed.

"out" FOLDER:
- This is where the classification reports, learning curve plots and saved models will be placed.

"utils" FOLDER:
- This folder should include all utility scripts used by the main script.

## ------ SCRIPT USAGE ------
### Arguments for CNN_Class.py script:
**Required**
Argument         | What it specifies / does
---------------- | -------------------------
"-l" / "--label" | What label you want to use. 'Race' or 'Age'.
"-i" / "--input_data" | What you want to use for the classification. 'Statement' or 'Sentiment'.
"-c" / "--class_type" | What type of classification you want to use. Logistical regression ('LogReg') or a simple neural network ('NN').

**Optional**
Argument         | What it specifies / does
---------------- | -------------------------
"-s" / "--save_model" | What you want the model to be saved as. Do **not** give af file extension! If no argument is given, the model won't save.

## ------ RESULTS ------
The scripts achieve what they set out to do. However, there are glaring problems. In short: It is a model that tries to predict something that seemingly cannot be predicted! This means that models are not actually that useful. 

Ex. the classification scores for age and statement are around 0.5. Seeing as there are only two choices, the models has not better scores that if it classified them by flipping a coin. The same can be seen when using race. 

Another flaw of the model / script is it's inability to balance the dataset when using sentiment. This means the model has way to many "black" and "white" inmates compared to hispanic, which in turn means the model gives scores around 0 when predicting hispanic. This is both due to a flaw in the script, but again also points towards the fact that you cannot predict age or race from last words.

In other words: DO NOT TRY TO USE THIS MODEL TO ACTUALLY PREDICT RACE OR AGE FROM DEATH ROW INMATES LAST STATEMENTS!

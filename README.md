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
- Save a learning curve plot to visualise the cross-validation of the model.
- If specified the script can also save the model for later use.

## ------ DATA ------
The data is a list of last statements made by deathrow inmates in Texas from 1982-2022. 

The data can be found here: https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html

However, as the data cannot be downloaded from the website, a webscraping script is needed. This can be done by the user or by using the R-markdown script "Webscraping_statements.rmd" created by myself.

If webscraped by the user, the following in the column "Last Statement" need to be transformed to "NA":  

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py script.

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
The scripts achieve what they set out to do. The classification reports also show a clear improvement when using a CNN model compared to a LogReg model. As the time it took to run the CNN model was not much longer than the LogReg model, it would in almost all cases be better to use that model for relevant predictions. 

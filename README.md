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
The data is a .csv file 28643 comments with the labels "0" or "1". As indicated by the reasearch papers that created the dataset. "0" is non-threating comments and "1" is threatning comments.

The data was obtained through the language analytics course.

The research papers behind the data:
- Hammer, H. L., Riegler, M. A., Øvrelid, L. & Veldal, E. (2019). "THREAT: A Large Annotated Corpus for Detection of Violent Threats". 7th IEEE International Workshop on Content-Based Multimedia Indexing.
- Wester, A. L., Øvrelid, L., Velldal, E., & Hammer, H. L. (2016). "Threat detection in online discussions". Proceedings of the 7th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis.

## ------ REPO STRUCTURE ------
"src" FOLDER:
- This folder contains the .py scripts.

"in" FOLDER:
- This is where the data used in the scripts should be placed. Ie. where the "VideoCommentsThreatCorpus.csv" should be placed.

"out" FOLDER:
- This is where the classification reports will be placed.

"utils" FOLDER:
- This folder should include all utility scripts used by the main script.

## ------ SCRIPT USAGE ------
### Arguments for LogReg_Class.py script:
**Required**
Argument         | What it specifies / does
---------------- | -------------------------
"-v" / "--vector" | What type of vectorizer you want to use. Either 'count' or 'tfidf'?
"-c" / "--class_name" | What you want your classification report to be named. Remember to include '.txt'


### Arguments for CNN_Class.py script:
Argument         | What it specifies / does
---------------- | -------------------------
"-c" / "--class_name" | What you want your classification report to be named. Remember to include '.txt'

**Optional**
Argument         | What it specifies / does
---------------- | -------------------------
"-s" / "--seq_length" | How big you want the sequence length to be. 750 if not specified.
"-em" / "--embed_size" | How big you want the embed size to be. 300 if not specified. 
"-e" / "--epoch" | How many epochs you want the model to run. 5 if not specified.
"-b" / "--batch" | What batchsize you want the model to use. 128 if not specified. 

## ------ RESULTS ------
The scripts achieve what they set out to do. The classification reports also show a clear improvement when using a CNN model compared to a LogReg model. As the time it took to run the CNN model was not much longer than the LogReg model, it would in almost all cases be better to use that model for relevant predictions. 

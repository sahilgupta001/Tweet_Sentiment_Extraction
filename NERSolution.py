import random
from spacy.util import compounding
from spacy.util import minibatch
import pandas as pd
import numpy as np
import os
import spacy
from tqdm import tqdm

df_train = pd.read_csv("./train.csv")
df_test = pd.read_csv("./test.csv")
df_submission = pd.read_csv("./submission.csv")


# Calculating the number of words in the main file in the main text
df_train['num_of_words'] = df_train['text'].apply(lambda x: len(str(x).split()))

# Kepping only the tweets that have length greater than 3
df_train = df_train[df_train['num_of_words'] > 3]


#Function to save the model
def save_model(model_path, nlp, model_name):
    output_dir = f'./{model_path}/'
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        nlp.meta["name"] = model_name
        nlp.to_disk(output_dir)
        print("The model has been saved")


#Fetching the model path
def model_out_path(sentiment):
    model_out_path = None
    if sentiment == 'positive':
        model_path = "models/model_pos"
    if sentiment == 'negative':
        model_path = "models/model_negative"
    return model_path


# Function to return thee training data to be used to train the NER model by Spacy
def get_training_data(sentiment):
    train_data = []
    for index, row in df_train.iterrows():
        if row.sentiment == sentiment:
            selected_text = row.selected_text
            text = row.text
            start = text.find(selected_text)
            end = start + len(selected_text)
            train_data.append((text, {"entities": [[start, end, 'selected_text']]}))

    return train_data


# Defining the model to be trained

def train(train_data, model_path, n_iter, model = None):
    #   We will firstly try to load the model and will resume training if it is found
    if model is not None:
        nlp = spacy.load(model_path)
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")
        print("Created blank 'en' model")
    # Create the built-in pipeline components and add them to the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last = True)
    else:
        ner = nlp.get_pipe("ner")

    for _ , annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # fetching the names of the other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        if model is None:
            nlp.begin_training()
        else:
            nlp.resume_training()

        for itn in tqdm(range(n_iter)):
            random.shuffle(train_data)
            batches = minibatch(train_data, size = compounding(4.0, 500.0, 1.001))
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts,  # batch of texts
                           annotations,  # batch of annotations
                           drop=0.5,  # dropout - make it harder to memorise data
                           losses=losses,
                           )
            print("Losses", losses)
    save_model(model_path, nlp, 'st_ner')


# Now we are training the model for positive and negative tweets
sentiment = 'positive'
train_data = get_training_data(sentiment)
model_path = model_out_path(sentiment)

# Training the model for the positive sentiment tweets
train(train_data, model_path, n_iter = 10, model = None)


sentiment = 'negative'
train_data = get_training_data(sentiment)
model_path = model_out_path(sentiment)

# Training the model for the negative sentiment tweets
train(train_data, model_path, n_iter = 10, model = None)


# Predicting the said model
def predict_entities(text, model):
    doc = model(text)
    ent_array = []
    for ent in doc.ents:
        start = text.find(ent.text)
        end = start + len(ent.text)
        new_int = [start, end , ent.label_]
        if new_int not in ent_array:
            ent_array.append([start, end, ent.label_])
    selected_text = text[ent_array[0][0]: ent_array[0][1]] if len(ent_array) > 0 else text
    return selected_text

selected_texts = []
trained_models_path = "./models/"

if trained_models_path is not None:
    print("Loading the models from ", trained_models_path)
    model_pos = spacy.load(trained_models_path + "model_pos")
    model_neg = spacy.load(trained_models_path + "model_negative")

    for index, row in df_test.iterrows():
        text = row.text
        output_str = ""
        if row.sentiment == 'neutral' or len(text.split()) <= 2:
            selected_texts.append(text)
        elif row.sentiment == 'positive':
            selected_texts.append(predict_entities(text, model_pos))
        elif row.sentiment == 'negative':
            selected_texts.append(predict_entities(text, model_neg))

df_test['selected_text'] = selected_texts

df_submission['selected_text'] = df_test['selected_text']
df_submission.to_csv("submission.csv", index=False)
print(df_submission.head(10))
import string
from itertools import count

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
train = train.dropna()

# Cleaning the data
train['text'] = train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())


# Splitting the data in train and test set
X_train, X_val = train_test_split(
    train, train_size = 0.80, random_state = 0)


# Making the training data to split in the category wise data
pos_data = X_train[X_train['sentiment'] == 'positive']
neg_data = X_train[X_train['sentiment'] == 'negative']
neutral_data = X_train[X_train['sentiment'] == 'neutral']

# Preparing the list of all the words occurring a particular sentiment class
train["words_list"] = train["text"].apply(lambda x: x.split())
pos_words = [item for sublist in train["words_list"][train["sentiment"] == "positive"] for item in sublist]
neg_words = [item for sublist in train["words_list"][train["sentiment"] == "negative"] for item in sublist]
neutral_words = [item for sublist in train["words_list"][train["sentiment"] == "neutral"] for item in sublist]

cv = CountVectorizer(lowercase= True, max_df=0.95, min_df=2,
                                     max_features=10000, stop_words = "english")

final_cv = cv.fit_transform(train['text'])
pos_cv = pd.DataFrame(cv.transform(pos_data['text']).toarray(), columns = cv.get_feature_names())
neg_cv = pd.DataFrame(cv.transform(neg_data['text']).toarray(), columns = cv.get_feature_names())
neutral_cv = pd.DataFrame(cv.transform(neutral_data['text']).toarray(), columns = cv.get_feature_names())


# Creating the dictionaries of the words within each sentiment where values are the proportions of the words tweets that contain that word
pos_dict = {}
neg_dict = {}
neutral_dict = {}

for k in cv.get_feature_names():
    pos = pos_cv[k].sum()
    neg = neg_cv[k].sum()
    neutral = neutral_cv[k].sum()

    pos_dict[k] = pos / pos_data.shape[0]
    neg_dict[k] = neg / neg_data.shape[0]
    neutral_dict[k] = neutral / neutral_data.shape[0]

    pos_dict[k] = pos_dict[k] - (neg_dict[k] + neutral_dict[k])
    neg_dict[k] = neg_dict[k] - (pos_dict[k] + neutral_dict[k])
    neutral_dict[k] = neutral_dict[k] - (pos_dict[k] + neg_dict[k])


# Calculating the selected text

def calculate_selected_text(tweet, sentiment, tol = 0):
    if (sentiment == 'neutral'):
        return tweet
    elif (sentiment == 'positive'):
        dict_to_use = pos_dict
    elif (sentiment == 'negative'):
        dict_to_use = neg_dict

    words = tweet.split()
    word_len = len(words)
    subsets = [words[i:j+1] for i in range(word_len) for j in range(i, word_len)]
    score = 0
    selection_string = ""
    lst = sorted(subsets, key = len)
    for i in range(len(subsets)):
        new_score = 0
        for p in range(len(lst[i])):
            if (lst[i][p].translate(str.maketrans('', '', string.punctuation)) in dict_to_use.keys()):
                new_sum = dict_to_use[lst[i][p] .translate(str.maketrans('', '', string.punctuation))]
                if (new_sum > score + tol):
                    score = new_sum
                    selection_string = lst[i]

    if (len(selection_string) == 0):
        return tweet

    return ' '.join(selection_string)


# Sending the data for the prediction
tol = 0.001
X_val['predicted_selection'] = ''

for index, row in X_val.iterrows():
    selected_text = calculate_selected_text(row['text'], row['sentiment'], tol)
    X_val[X_val["textID"] == row["textID"]]["predicted_selection"] = selected_text


# Generating the submission for the sample file

sample = pd.read_csv("./sample_submission.csv")
for index, row in test.iterrows():
    selected_text = calculate_selected_text(row['text'], row['sentiment'], tol)
    print(row['textID'], selected_text)
    sample.loc[sample['textID'] == row['textID'], ['selected_text']] =  selected_text

sample.to_csv('./submission.csv', index = False)

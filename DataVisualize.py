import csv
import string
import nltk
from nltk.corpus import stopwords
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re


train_path = "./train.csv"
train = pd.read_csv(train_path)
train = train.dropna(how = 'any', axis = 0)

# def data_visualize():
#     # printing the description of the data
#     print("The description of the data is ...")
#     print(train.describe())
#     # Printing the counts of all the sentiment classes
#     print("Counting the number of occurences of each sentiment class...")
#     temp = train.groupby('sentiment').count()['text'].reset_index().sort_values(by = "text", ascending = False)
#     # visualizing the count data
#     plt.figure()
#     sns.countplot(x = 'sentiment', data = train)
#     plt.show()
#
# data_visualize()


#  Measuring the jaccaard similarity
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)/ len(a) + len(b) - len(c))

results = []
for ind, row in train.iterrows():
    jaccard_score = jaccard(row.text, row.selected_text)
    results.append((row.text, row.selected_text, jaccard_score))

jaccard = pd.DataFrame(results, columns = ["text", "selected_text", "jaccard_score"])
train = pd.merge(train, jaccard, how = "outer")
train['number_words_text'] = train['text'].apply(lambda x: len(str(x).split()))
train['number_words_ST'] = train['selected_text'].apply(lambda x: len(str(x).split()))
train['difference_in_words'] = train['number_words_text'] - train['number_words_ST']

# visualising the data with the class and the difference in the number of words
# plt.figure()
# fig = sns.distplot(train['number_words_ST'])
# fig = sns.distplot(train['number_words_text'])
# plt.show()

# Visualising the kernel distribution of the number of words in the selected text and the normal tweet text

# plt.figure()
# fig = sns.kdeplot(train['number_words_ST'], shade = True, color = 'r')
# fig = sns.kdeplot(train['number_words_text'], shade = True, color = 'g')
# fig.set_title('Kernel Distribution Plot of Number of Words')
# plt.show()


# visualisng the difference in number of words and jaccard scores across different sentiments
# plt.figure()
# fig = sns.kdeplot(train[train['sentiment'] == 'positive']['difference_in_words'], color = 'r', shade = True)
# fig = sns.kdeplot(train[train['sentiment'] == 'negative']['difference_in_words'], color = 'b', shade = True)
# plt.show()

# Visualising the jaccard score of the various tweets
# plt.figure()
# fig = sns.kdeplot(train[train['sentiment'] == 'positive']['jaccard_score'], color = 'r', shade = True)
# fig = sns.kdeplot(train[train['sentiment'] == 'negative']['jaccard_score'], color = 'b', shade = True)
# plt.legend(labels=['positive','negative'])
# plt.show()



# Vusualising the tweets that have less than two words in the tweets
# tweets = train[train['number_words_text'] <= 2]
# print(tweets.groupby('sentiment').count())



# Cleaning the corpus
# def clean_data(text):
#     text = str(text).lower()
#     text = re.sub('\[.*?\]', '', text)
#     text = re.sub('https?://\S+|www\.\S+', '', text)
#     text = re.sub('<.*?>+', '', text)
#     text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
#     text = re.sub('\n', '', text)
#     text = re.sub('\w*\d\w*', '', text)
#     return text
#
#
# train['text'] = train['text'].apply(lambda x: clean_data(x))
# train['selected_text'] = train['selected_text'].apply(lambda x: clean_data(x))
# print(train.head())


# Trying to visualise the most common words in the selected text


def remove_stopword(text):
    return [y for y in text if y not in  stopwords.words('english')]

train['temp_list'] = train['selected_text'].apply(lambda x: str(x).split())
train['temp_list'] = train['temp_list'].apply(lambda x: remove_stopword(x))
top = nltk.Counter([item for sublist in train['temp_list'] for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words', 'count']
temp.style.background_gradient(cmap = 'Blues')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tokenizer as tokenizer
#from transformers.models.pop2piano.convert_pop2piano_weights_to_hf import model

plt.style.use('ggplot')

import nltk
#Make sure to download the below packages. The comment the lines below, so you don't download those packages each time.
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')

#Read the data
df = pd.read_csv('Reviews.csv')
print(df.shape)
#For this example, let's analyse only the first 100 rows, for a faster processing
df = df.head(100)
print(df.shape)

df.head()

#Let's perform a quick EDA
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

#Basic NLTK
example = df['Text'][50]
print(example)

tokens = nltk.word_tokenize(example)
#tokens = nltk.tokenize.word_tokenize(example, language='english',preserve_line=False)
tokens[:10]

tagged = nltk.pos_tag(tokens)
tagged[:10]

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


#We shall start with the simpler VADER Sentiment Score
from nltk.sentiment import SentimentIntensityAnalyzer
#from nltk import sentiment as vader
from tqdm.notebook import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia

#Let's run a quick test
sentences=['I am so happy!','This is the worst thing ever']
sid = sia()
for sentence in sentences:
    ss = sid.polarity_scores(sentence)


#sia.polarity_scores(example)
sentences=[example]
sid = sia()
for sentence in sentences:
    ss = sid.polarity_scores(sentence)

#Run the polarity score on the entire dataset

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    texts = row['Text']
    myid = row['Id']
    for text in texts:
        res[myid] = sid.polarity_scores(texts)

#Let's load and transpose the dataftame with .T
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

vaders.head()

#Let's plot the results
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

#Let's now try with the more powerful Roberta model from Hugging Face
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#Let's compare VADER's results with Roberta's
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}

print(scores_dict)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    #here we use try so the model doesn't break up if the reviews are too long
    try:
        texts = row['Text']
        myid = row['Id']
        for text in texts:
            vader_result = sid.polarity_scores(texts)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f"Broke for id {myid}")

    results_df = pd.DataFrame(res).T
    results_df = results_df.reset_index().rename(columns={'index': 'Id'})
    results_df = results_df.merge(df, how='left')

    #let's compare the scores between the two model

    results_df.columns


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos', 'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',
             palette='tab10')
plt.show()


#However, both models don't perform well with sarcasm
#Let's find when the model identifies positvie sentiment in 1 star reviews
results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]

#Let's find when the model identifies negative sentiment in 5 star reviews
results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]


#Alternatively, we can create a Transformers Pipeline to ease the job

from transformers import pipeline
sent_pipeline = pipeline("sentiment-analysis")

sent_pipeline('Sunny days are super cool!')
sent_pipeline('Rain is dull but the crops need it sometimes.')
sent_pipeline('Wow')
sent_pipeline('Woooow')
sent_pipeline('Boooooo')








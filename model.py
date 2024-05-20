import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.utils import resample
import string
import re
import nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from wordcloud import WordCloud
from collections import Counter

from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense,Flatten,Dropout
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

nltk.data.path.append("/Users/csawant/Desktop/chaitali_wrkspc/email_spam_tensorflow/nltk")

#importing the dataset
df=pd.read_csv("/Users/csawant/Desktop/chaitali_wrkspc/email_spam_tensorflow/SPAM text message 20170820 - Data.csv")
print(df)

#Printing the message attribute
print(df["Message"])

#checking the different categories in our email messages
print(df["Category"].value_counts())

#plotting the categories 

plt.figure(figsize=(6,6))
colors = ['#ff9999', '#66b3ff']
explode = (0, 0.1)
plt.pie(x=df['Category'].value_counts().values, labels=['ham', 'spam'],
        autopct='%.1f%%', startangle=70, explode=explode, colors=colors, shadow=True)
plt.legend(title="Category", loc="upper right")
plt.title("Category Distribution")
plt.show()

#balancing the data
ham_msg = df[df.Category == "ham"]
spam_msg = df[df.Category == 'spam']

ham_downsample = resample(ham_msg,
             replace=True,
             n_samples=len(spam_msg),
             random_state=42)

data = pd.concat([ham_downsample, spam_msg])
print(data)

#checking the different categories in our email messages
print(data["Category"].value_counts())

#data preprocessing
#lowercase messages
print(data["Message"].iloc[2])
def convert_lowercase(text):
    text = text.lower()
    return text

data['Message'] = data['Message'].apply(convert_lowercase)

print(data["Message"].iloc[2])

#removing url

print(data["Message"].iloc[8])
def remove_url(text):
    re_url = re.compile('https?://\S+|www\.\S+')
    return re_url.sub('', text)

data['Message'] = data['Message'].apply(remove_url)
print(data["Message"].iloc[8])

#removing all punctuations

print(data["Message"].iloc[6])
punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

data['Message']= data['Message'].apply(lambda x: remove_punctuations(x))
print(data["Message"].iloc[6])


#removing stopwords
print(data["Message"].iloc[12])
def remove_stopwords(text):
    stop_words = stopwords.words('english')

    imp_words = []

    # Storing the important words
    for word in str(text).split():
        word = word.lower()

        if word not in stop_words:
            imp_words.append(word)

    output = " ".join(imp_words)

    return output


data['Message'] = data['Message'].apply(lambda text: remove_stopwords(text))
print(data["Message"].iloc[12])

#removing all digits
print(data["Message"].iloc[-5])
data['Message'] = data['Message'].apply(lambda x:re.sub('[\d]','',x))
print(data["Message"].iloc[-5])

#stemming or lemmatization
print(data["Message"].iloc[2])
def perform_stemming(text):
    stemmer = PorterStemmer()
    new_list = []
    words = word_tokenize(text)
    for word in words:
        new_list.append(stemmer.stem(word))

    return " ".join(new_list)

data['Message'] = data['Message'].apply(perform_stemming)
print(data["Message"].iloc[2])

#frequent words in spam email
all_spam_words = []
for sentence in data[data['Category'] == "spam"]['Message'].to_list():
    for word in sentence.split():
        all_spam_words.append(word)
df = pd.DataFrame(Counter(all_spam_words).most_common(50), columns= ['Word', 'Frequency'])
df.style.background_gradient(cmap='Purples')

#tree of most common words
fig = px.treemap(df, path=['Word'], values='Frequency', title='Tree of Most Common Words', color='Frequency')
fig.show()

fig = px.bar(
    df,
    x="Frequency",
    y="Word",
    title='Common Words in SPAM',
    width=700,
    height=700,
    color='Frequency',  # Set the color based on the count of common words
    color_continuous_scale='YlGnBu',  # Specify a color scale
)
fig.update_layout(
    xaxis_title="Frequency",
    yaxis_title=None,  # Remove the y-axis title
    title_font=dict(size=40),  # Increase title font size
    font=dict(size=20),  # Increase general text font size
)


fig.update_xaxes(showgrid=False)  # Hide x-axis grid lines
fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')  # Show and style y-axis grid lines

fig.show()

#wordcloud

text = " ".join(data[data['Category'] == 'spam']['Message'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 500, width = 800,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('WordCloud for spam emails', fontsize=30)
plt.axis('off')
plt.show()

#frequent words in ham emails
all_spam_words = []
for sentence in data[data['Category'] == "ham"]['Message'].to_list():
    for word in sentence.split():
        all_spam_words.append(word)
df = pd.DataFrame(Counter(all_spam_words).most_common(50), columns= ['Word', 'Frequency'])
df.style.background_gradient(cmap='Blues')

#tree of most common words
fig = px.treemap(df, path=['Word'], values='Frequency', title='Tree of Most Common Words', color='Word')
fig.show()

#most common negative words
fig = px.bar(
    df,
    x="Frequency",
    y="Word",
    title='Most Common Negative Words',
    width=700,
    height=700,
    color='Frequency',  # Set the color based on the count of common words
    color_continuous_scale='YlOrRd',  # Choose the 'YlGnBu' color scale
)

fig.show()

#wordcloud for ham words
text = " ".join(data[data['Category'] == 'ham']['Message'])
plt.figure(figsize = (15, 10))
wordcloud = WordCloud(max_words=500, height= 500, width = 800,  background_color="black", colormap= 'viridis').generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.title('WordCloud for ham emails', fontsize=30)
plt.axis('off')
plt.show()

#max number of words in text
from collections import Counter

def counter_word(text_col):
    count = Counter()
    for text in text_col.values:
        for word in text.split():
            count[word] += 1
    return count


counter = counter_word(data.Message)

MAX_NB_WORDS = len(counter)
print(MAX_NB_WORDS)

data['Message_Length'] = data['Message'].apply(len)

MAX_LEN = data['Message_Length'].max()
print(MAX_LEN)


#tokenizing
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data['Message'].values)

word_index=tokenizer.word_index

import math

vocab_size = len(word_index)  # Assuming you have a word index from your dataset
embedding_dim = int(math.sqrt(vocab_size) / 2)  # Adjust the constant factor as needed

print(vocab_size)
print(embedding_dim)

#Modeling
from sklearn import preprocessing
labelencoder = preprocessing.LabelEncoder()
data['Category'] = labelencoder.fit_transform(data['Category'])
Y = data['Category'].values
print('Shape of label tensor:', Y.shape)

X = tokenizer.texts_to_sequences(data['Message'].values)
X = pad_sequences(X, maxlen=MAX_LEN)
print('Shape of data tensor:', X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=30000,output_dim = 30, input_length=X_train.shape[1] ))
model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# Train the model
epochs = 10
history=model.fit(X_train, Y_train ,validation_data=(X_test, Y_test),epochs=epochs,callbacks=[early_stop],verbose=2)

model.evaluate(X_test, Y_test)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
epoch = [i + 1 for i in range(len(train_acc))]

acc_loss_df = pd.DataFrame({"Training Loss" : train_loss,
                            "Validation Loss": val_loss,
                            "Train Accuracy" : train_acc,
                            "Validation Accuracy" : val_acc,
                            "Epoch":epoch})


acc_loss_df.style.bar()

fig = go.Figure()

fig.add_trace(go.Scatter(x = acc_loss_df['Epoch'],
                         y = acc_loss_df['Train Accuracy'],
                         mode='lines+markers',
                         name='Training Accuracy'))

fig.add_trace(go.Scatter(x = acc_loss_df['Epoch'],
                         y = acc_loss_df['Validation Accuracy'],
                         mode='lines+markers',
                         name = 'Validation Accuracy'))

fig.update_layout(title = {'text': "<b>Training Accuracy Vs Validation Accuracy</b>\n",
                           'xanchor': 'center',
                           'yanchor': 'top',
                           'y':0.9,'x':0.5,},
                  xaxis_title="Epoch",
                  yaxis_title = "Accuracy",
                  title_font = dict(size = 20))

fig.layout.template = 'plotly_dark'

fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x = acc_loss_df['Epoch'],
                         y = acc_loss_df['Training Loss'],
                         mode='lines+markers',
                         name='Training Loss'))

fig.add_trace(go.Scatter(x = acc_loss_df['Epoch'],
                         y = acc_loss_df['Validation Loss'],
                         mode='lines+markers',
                         name = 'Validation Loss'))

fig.update_layout(title = {'text': "<b>Training Loss Vs Validation Loss</b>\n",
                           'xanchor': 'center',
                           'yanchor': 'top',
                           'y':0.9,'x':0.5,},
                  xaxis_title="Epoch",
                  yaxis_title = "Loss",
                  title_font = dict(size = 20))

fig.layout.template = 'plotly_dark'

fig.show()

threshold = 0.5

result = model.predict(X_test, verbose=2)
result = result > threshold
result = result.astype("int32")

from sklearn.metrics import classification_report
target_names = ['spam','ham']
print(classification_report(Y_test, result, target_names=target_names))

cm = confusion_matrix(Y_test,result)
classes = ['spam','ham']

plt.figure(figsize = (3,3))
sns.heatmap(cm, annot = True, fmt = 'd', cbar=False).set(xticklabels = classes, yticklabels = classes)
plt.xlabel("\nPrediction", size = 15)
plt.ylabel("\nTruth",  size = 15)
plt.show()
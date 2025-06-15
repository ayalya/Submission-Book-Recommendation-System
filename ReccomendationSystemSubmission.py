import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras import layers
from pathlib import Path
import re

!pip install kagglehub
import kagglehub

# Download Dataset latest version
path = kagglehub.dataset_download("arashnic/book-recommendation-dataset")

books_path = os.path.join(path, 'Books.csv')
ratings_path = os.path.join(path, 'Ratings.csv')
users = os.path.join(path, 'Users.csv')

books = pd.read_csv(books_path, encoding='utf-8')
ratings = pd.read_csv(ratings_path, encoding='utf-8')
users = pd.read_csv(users, encoding='utf-8')

books = books.drop(['Image-URL-S', 'Image-URL-M', 'Image-URL-L'], axis=1)

# Preprocessing
rat_mod = ratings.groupby('ISBN')['Book-Rating'].agg(lambda x: x.mode().iloc[0])

books_rating = pd.merge(books[['ISBN', 'Book-Title', 'Book-Author','Publisher', 'Year-Of-Publication']],
                        rat_mod, # Add the ratings DataFrame here
                        on='ISBN', how='left')

# Data Preparation Content-Based Filtering
books_rating = books_rating.dropna()

# Hapus salah input data
temp = books_rating[(books_rating['Year-Of-Publication'] == 'DK Publishing Inc') | (books_rating['Year-Of-Publication'] == 'Gallimard')]
books_rating = books_rating.drop(temp.index)
books_rating['Year-Of-Publication'].astype('int64')

# Standarisasi ISBN
clean_data = books_rating.sort_values(by='ISBN')
def clean_isbn(isbn):
   if pd.isnull(isbn):
       return None
   match = re.match(r'^(\d+)', str(isbn))
   if match:
        cleaned = match.group(1)
   if len(cleaned) == 10 or len(cleaned) == 13: # Mengambil ISBN dengan panjang 10 hingga 13
      return cleaned
   return None

clean_data['ISBN_clean'] = clean_data['ISBN'].apply(clean_isbn)

isbn_id = clean_data['ISBN_clean'].tolist()
book_title_id = clean_data['Book-Title'].tolist()
book_author_id = clean_data['Book-Author'].tolist()
publisher_id = clean_data['Publisher'].tolist()
year_id = clean_data['Year-Of-Publication'].tolist()
rating_id = clean_data['Book-Rating'].tolist()

books_new = pd.DataFrame(
    {'ISBN': isbn_id,
     'Book-Title': book_title_id,
     'Book-Author': book_author_id,
     'Publisher': publisher_id,
     'Year-Of-Publication': year_id}
)

books_new = books_new[:20000]
df_cbf = books_new.copy()

# TF-IDF
# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

tfidf.fit(df_cbf['Book-Author'])

tfidf.get_feature_names_out()

tfidf_matrix = tfidf.fit_transform(df_cbf['Book-Author'])
tfidf_matrix.todense()

# Cosine Similarity
cosine = cosine_similarity(tfidf_matrix)

# Data Preparation Collaborative Filtering
df_rating = ratings[ratings['Book-Rating']>0].copy()
df_rating = df_rating[:20000]

# Encoding user dan ISBN
user_ids = df_rating['User-ID'].unique().tolist()
user_to_user_encoder = {x: i for i, x in enumerate(user_ids)}
user_encoder_to_user = {i: x for i, x in enumerate(user_ids)}

isbn_id = df_rating['ISBN'].unique().tolist()
isbn_to_isbn_encoder = {x: i for i, x in enumerate(isbn_id)}
isbn_encoder_to_isbn = {i: x for i, x in enumerate(isbn_id)}

pd.options.mode.chained_assignment = None

# Maping User-ID ke user pada DataFrame
df_rating['user'] = df_rating['User-ID'].map(user_to_user_encoder)
# Maping ISBN ke judul buku pada DataFrame
df_rating['book_title'] = df_rating['ISBN'].map(isbn_to_isbn_encoder)


min_rating = min(df_rating['Book-Rating'])
max_rating = max(df_rating['Book-Rating'])

# Pemisahan data
cf_df = df_rating.sample(frac=1, random_state=42)

x = cf_df[['user', 'book_title']].values
y = cf_df['Book-Rating'].apply(lambda x: (x - min_rating) /
 (max_rating - min_rating)).values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# MODEL DEVELOPMENT
# CONTENT-BASED FILTER
# Convert the cosine similarity matrix to a pandas DataFrame
cosine_df = pd.DataFrame(cosine, index=df_cbf['Book-Title'], columns=df_cbf['Book-Title'])

feature_cbf = df_cbf[['Book-Title', 'Book-Author']]
def cbf_recommendation(book_title, similarity_data=cosine_df, dataset=feature_cbf, k=5):
  index = similarity_data.loc[:,book_title].to_numpy().argpartition(
      range(-1, -k, -1))
  closest = similarity_data.columns[index[-1:-(k+2):-1]]
  closest = closest.drop(book_title, errors='ignore')
  return pd.DataFrame(closest).merge(dataset).head(k)

# Predict
book_title_test = "The Woman in the Moon and Other Tales of Forgotten Heroines" # book title example

df_cbf[df_cbf['Book-Title'].eq(book_title_test)]

cbf_recommendation(book_title_test)

# COLLABORATIVE FILTERING

class RecommederNet(tf.keras.Model):
      # Fungsi inisialisasi
  def __init__(self, num_user, num_title, embedding_size, dropout_rate=0.3, **kwargs):
    super(RecommederNet, self).__init__(**kwargs)
    self.num_user = num_user
    self.num_title = num_title
    self.embedding_size = embedding_size
    self.dropout_rate = dropout_rate

    self.user_embedding = tf.keras.layers.Embedding(
        num_user,
        embedding_size,
        embeddings_initializer="he_normal",
        embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
    )
    self.user_bias = tf.keras.layers.Embedding(num_user, 1) # Layer embedding bias

    self.book_embedding = tf.keras.layers.Embedding(
        num_title,
        embedding_size,
        embeddings_initializer="he_normal",
        embeddings_regularizer=tf.keras.regularizers.l2(1e-6)
    )
    self.book_bias = tf.keras.layers.Embedding(num_title, 1) # Layer embedding bias

    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:, 0]) # Call embedding layer 1
    user_vector = self.dropout(user_vector)

    user_bias = self.user_bias(inputs[:, 0]) # Call embedding layer 2

    book_title_vector = self.book_embedding(inputs[:, 1]) # Call embedding layer 3
    book_title_vector = self.dropout(book_title_vector)

    book_title_bias = self.book_bias(inputs[:, 1]) # Call embedding layer 4
    dot_user_book = tf.tensordot(user_vector, book_title_vector, 2) # Dot product

    x = dot_user_book + user_bias + book_title_bias
    return tf.nn.sigmoid(x)
  
model = RecommederNet(num_user, num_title, 50) # inisialisasi model

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanSquaredError()]
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=1,
    mode='min'
)
hitsory = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=50,
    verbose=1,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

books_df = clean_data

# Pilih user secara acak
user_id = df_rating['User-ID'].sample(1).iloc[0]
book_read_by_user = df_rating[df_rating['User-ID'] == user_id]

# Membuat variabel book_not_readed
book_not_readed = books_df[~books_df['ISBN'].isin(book_read_by_user['ISBN'].values)]['ISBN']
book_not_readed = list(set(book_not_readed).intersection(set(isbn_to_isbn_encoder.keys())))

book_not_readed = [[isbn_to_isbn_encoder.get(x)] for x in book_not_readed]
user_encoder = user_to_user_encoder.get(user_id)
user_book_array = np.hstack(([[user_encoder]] * len(book_not_readed), book_not_readed))

# Predict
ratings_model = model.predict(user_book_array).flatten()

top_rating_indicates = ratings_model.argsort()[-10:][::-1]
top_rating_books = [isbn_encoder_to_isbn.get(book_not_readed[x][0]) for x
                    in top_rating_indicates]

top_book_user = (
    book_read_by_user.sort_values(by='Book-Rating', ascending=False)
    .head(5)
    .ISBN.values
)

# Get the DataFrame rows for the recommended books and the user's highly rated books
book_df_rows = books_df[books_df['ISBN'].isin(top_rating_books + list(top_book_user))].copy()

# Select the relevant columns and rename them for the recommended books output
df_recommended_books = book_df_rows[book_df_rows['ISBN'].isin(top_rating_books)][['Book-Title', 'Book-Author', 'Publisher']].copy()
df_recommended_books.columns = ['Book Title', 'Book Author', 'Publisher']


# Select the relevant columns and rename them for the user's highly rated books output
df_book_readed_by_user = book_df_rows[book_df_rows['ISBN'].isin(top_book_user)][['Book-Title', 'Book-Author']].copy()
df_book_readed_by_user.columns = ['Book Title', 'Book Author']


# Displays recommendation results in DataFrame form
print("Showing recommendation for users: {}".format(user_id))
print("===" * 9)
print("Book with high ratings from user")
print("----" * 8)
print(df_book_readed_by_user)
print("----" * 8)
print("Top 10 books recommendation")
print("----" * 8)
df_recommended_books

# EVALUASI MODEL
# Content Based
threshold = 0.5

ground_truth = (cosine >= threshold).astype(int)

ground_truth_df = pd.DataFrame(ground_truth, index=df_cbf['Book-Title'],
                               columns=df_cbf['Book-Title']).sample(5, axis=1)

# Ambil proposisi dari cosine cimilarity dan ground truth matrix
sample_size = 1000
cosine_sample = cosine[:sample_size, :sample_size]
ground_truth_sample = ground_truth[:sample_size, :sample_size]

# Convert cosine similarity dan ground truth ke dalam array 1D
cosine_flat = cosine_sample.flatten()
ground_truth_flat = ground_truth_sample.flatten()

# Kalkulasi Precision, Recall, dan F1 Score
precision = precision_score(ground_truth_flat, (cosine_flat >= threshold).astype(int), zero_division=1)
recall = recall_score(ground_truth_flat, (cosine_flat >= threshold).astype(int), zero_division=1)
f1 = f1_score(ground_truth_flat, (cosine_flat >= threshold).astype(int), zero_division=1)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# COLLABORATIVE FILTERING
# Evaluasi train
train_loss, train_rmse, train_mse = model.evaluate(x_train, y_train, verbose=0)
test_loss, test_rmse, test_mse = model.evaluate(x_test, y_test, verbose=0)
print(f"Train Loss: {train_loss:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Train MSE: {train_mse:.4f}")
print('')
print(f"Test Loss: {test_loss:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
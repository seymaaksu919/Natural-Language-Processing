import nltk
import string
import re
from textblob import TextBlob
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

text = "John was running       quickly to   catcch   the bus, but he missed it anyway. He runs every morning to stay healthy. Better habits make a better life!"

# Boşlukları temizle
cleaned_text1 = " ".join(text.split())

# Küçük harf
cleaned_text2 = cleaned_text1.lower()

# Noktalama kaldır
cleaned_text3 = cleaned_text2.translate(str.maketrans("", "", string.punctuation))

# Özel karakter kaldır (bu örnekte gerek yok ama örnek olsun)
cleaned_text4 = re.sub(r"[^A-Za-z0-9\s]", "", cleaned_text3)

# Yazım düzelt
cleaned_text5 = str(TextBlob(cleaned_text4).correct())

# Tokenize et
world_token = nltk.word_tokenize(cleaned_text5)

# Stopwords kaldır
stop_words_eng = set(stopwords.words("english"))
filtered_words = [word for word in world_token if word.lower() not in stop_words_eng]

# Stemming
stemm = PorterStemmer()
stems = [stemm.stem(w) for w in filtered_words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(w, pos="v") for w in filtered_words]

import pandas as pd

df = pd.DataFrame({
    "Word": filtered_words,
    "Stem": stems,
    "Lemma": lemmas
})

print(df)













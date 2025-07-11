import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

df = pd.read_csv("IMDB Dataset.csv")

df2 = df.head(100)



#Metin verilerini alalım

documents = df["review"]
labels = df["sentiment"]



#Metin önişleme adımları

def clean_text(text):
    
    text=text.lower()
    
    text=re.sub(r"[^A-Za-z0-9\s]","", text)
    
    text=" ".join([word for word in text.split()
                   if len (word)>2])
    
    
    return text

#Metinler temizlenir ve doc içine atılır.
cleaned_documents = [clean_text(doc) for doc in documents]    
   

#Bow adımları

vectorizer = CountVectorizer()

#Metinler sayısallaştırılır.
X = vectorizer.fit_transform(cleaned_documents[:100])


#kelime kümesi oluşturulur.
feature_name = vectorizer.get_feature_names_out()

#Vektör temsili için
print("Vektör temsili:" , X.toarray()[:2])


df_bow=pd.DataFrame(X.toarray(),columns=feature_name)

word_counts = X.sum(axis=0).A1
word_freq=dict(zip(feature_name,word_counts))


most_common_words= Counter(word_freq).most_common(5)
print(most_common_words)

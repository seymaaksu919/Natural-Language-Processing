from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd 
import numpy as np


#Örnek belge

documents = ["Kedi çok tatlı bir hayvandır." ,
             "kedi ve köpekler çok tatlı hayvanlardır",
             "Arılar bal üretirler"]

tfidf_vectorizer= TfidfVectorizer()

#Metinleri sayısala çeviririz.

X = tfidf_vectorizer.fit_transform(documents)

#kelime kümesi

feature_names = tfidf_vectorizer.get_feature_names_out()


print("TF-IDF Vektor temsilleri:")
vektor_temsili = X.toarray()
print(vektor_temsili)

#DataFrame haline getirilir.
df_tfidf = pd.DataFrame(vektor_temsili, columns = feature_names)


#Sadece kedi sütunun ortalamasına bakılır.
kedi_tfidf = df_tfidf["kedi"]
kedi_mean_tfidf = kedi_tfidf.mean()

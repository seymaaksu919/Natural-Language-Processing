from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("/home/seymaaksu/spyder/sms_spam.csv")

vectorizer = TfidfVectorizer()
X=vectorizer.fit_transform(df["text"])


#kelime kümesi alınır.

feature_names = vectorizer.get_feature_names_out()
tfidf_score = X.mean(axis=0).A1 # ortalama tf-idf degerleri


df_tfidf = pd.DataFrame({"word":feature_names, "tfidf_score":tfidf_score})
df_tfidf_sorted = df_tfidf.sort_values(by = "tfidf_score", ascending=False)

print(df_tfidf_sorted)

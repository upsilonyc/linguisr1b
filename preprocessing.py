import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# nltk.download("punkt_tab")
# nltk.download("wordnet")
# nltk.download("stopwords")
l = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# tokenize
def tokenizer(p):
    """
    "Hi you!" --> ["hi", "you"]
    """
    return word_tokenize(p)

# Lemmatize
def lemmatize(w):
    """
    "competing" --> "compete"
    """
    return l.lemmatize(w)

def lemmatize_count(p, res={}): # p is a raw paragraph
    """
    "Hi you!" --> ["hi", "you", "!"] --> ["hi", "you"] --> {"hi": 1, "you": 1}
    """
    q = tokenizer(p)
    for i in q:
        if i in ["!", ",", ".", ":", ";", "..."] or i in stop_words:
            q.remove(i)
    for i in q:
        ld = lemmatize(i)
        if ld in res:
            res[ld] += 1
        else:
            res[ld] = 1
    return res

# visualization
def bar_plot(d):
    labels, vals = list(d.keys()), list(d.values())
    plt.bar(labels, vals, color="grey")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title("Most Frequent Words in Technical Job Self-Representations")
    plt.xticks(rotation=30)
    plt.show()

if __name__ == "__main__":
    n = int(input("num of queries: "))
    res = {}
    for i in range(n):
        p = input("paragraph: ")
        res = lemmatize_count(p, res)
    limit = 10 if n >= 10 else n
    k = sorted(res, key=res.get, reverse=True)
    s = {key : res[key] for key in k}
    bar_plot(s)

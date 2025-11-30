import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# nltk.download("punkt_tab")
# nltk.download("wordnet")
# nltk.download("stopwords")
# unquote the lines above if running this program for the first time
l = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# tokenize
def tokenizer(p):
    """
    "Hi you!" --> ["hi", "you"]
    """
    return word_tokenize(p.lower())

# Lemmatize
def lemmatize(w):
    """
    "competing" --> "compete"
    """
    return l.lemmatize(w)

def lemmatize_count(p, res=None): # p is a raw paragraph
    """
    "Hi you!" --> ["hi", "you", "!"] --> ["hi", "you"] --> {"hi": 1, "you": 1}
    """
    q = tokenizer(p)
    junk = {"!", ",", ".", ":", ";", "...", "engineer", "full-stack", "frontend",
            "backend", "software", "year", "years", "python", "javascript",
            "develop", "development", "with", "5+", "application", "of", "solution",
            "design", "developing", "cloud", "platform", "ability", "project", "system",
            "deliver", "user", "web", "java", "react"} | stop_words
    q = [tok for tok in q if tok not in junk]
    for i in q:
        ld = lemmatize(i)
        if ld in res and ld not in junk:
            res[ld] += 1
        else:
            res[ld] = 1
    return res

def get_first_k_elements(dictionary, k):
    if not isinstance(dictionary, dict):
        raise TypeError("Input must be a dictionary.")
    if not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer.")

    # Convert the dictionary items to a list and slice it
    first_k_items = list(dictionary.items())[:k]

    # Convert the sliced list of tuples back into a dictionary
    return dict(first_k_items)

# visualization
def bar_plot(d):
    labels, vals = list(d.keys()), list(d.values())
    plt.bar(labels, vals, color="grey")
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.title("Most Frequent Words in Descriptions")
    plt.xticks(rotation=30)
    plt.show()

if __name__ == "__main__":
    n = int(input("num of queries: "))
    res = {}
    for i in range(n):
        p = input("paragraph: ")
        res = lemmatize_count(p, res)
    k = sorted(res, key=res.get, reverse=True)
    s = {key : res[key] for key in k}
    limit = 10 if len(s.keys()) > 10 else len(s.keys())
    bar_plot(get_first_k_elements(s, limit))

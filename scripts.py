import re
from bs4 import BeautifulSoup as bs
import pickle
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd


def download_nltk_resources():
    nltk.download('stopwords')


def predict(sentence):
    sentence_preprocessed = preprocess_sentence(sentence)
    model, tfidf_vectorizer, labels = deserialize_model()
    sentence_tfidf = tfidf_vectorizer.transform([sentence_preprocessed])
    prediction = model.predict(sentence_tfidf)[0]
    prediction_dataframe = pd.DataFrame(
        data={"IsPresent": prediction, "Label": labels})
    final_predictions = prediction_dataframe[
        prediction_dataframe["IsPresent"] == 1
        ]["Label"].values.tolist()
    return final_predictions


def deserialize_model():
    with open("model.pkl", "rb") as handle:
        model: OneVsRestClassifier = pickle.load(handle)
    with open("tfidf_vectorizer.pkl", "rb") as handle:
        tfidf_vectorizer = pickle.load(handle)
    with open("labels.pkl", "rb") as handle:
        labels = pickle.load(handle)
    return model, tfidf_vectorizer, labels


def preprocess_sentence(sentence):
    sentence = parse_html(sentence)
    sentence = tokenize(sentence)
    sentence = remove_stop_words(sentence)
    sentence = tag_pos(sentence)
    sentence = lemmatize(sentence, with_pos=True)
    sentence = stemmize(sentence)
    sentence = " ".join([str(item) for item in sentence])
    return sentence


def parse_html(cell):
    soup = bs(cell, "html.parser")

    script_tags = soup.find_all("script")
    for script_tag in script_tags:
        script_tag.extract()

    code_tags = soup.find_all("code")
    for code_tag in code_tags:
        code_tag.extract()

    preproc_cell = soup.get_text()
    preproc_cell = preproc_cell.replace(',', ' ')

    return preproc_cell


def tokenize(*texts):
    tokens = []
    for text in texts:
        # https://regex101.com/
        tokenizer = nltk.RegexpTokenizer(r'\.?[a-z#]+')
        tokens_temp = tokenizer.tokenize(text)
        tokens += [re.sub("(.)\\1{3,}", "\\1", token) for token in tokens_temp]
    return tokens


def remove_stop_words(cell):
    stop_words = stopwords.words("english")
    return [word for word in cell if word not in stop_words]


def _get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return 'n'


def tag_pos(cell):
    treebank_tags = pos_tag(cell)
    pos = [(tag[0], _get_wordnet_pos(tag[1])) for tag in treebank_tags]
    return pos


def lemmatize(cell, with_pos=False):
    lemmatizer = WordNetLemmatizer()
    if not with_pos:
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in cell]
    else:
        lemmatized_tokens = [lemmatizer.lemmatize(
            pair[0],
            pos=pair[1]
            ) for pair in cell]
    return lemmatized_tokens


def stemmize(cell):
    stemmer = SnowballStemmer("english")
    stemmed_tokens = [stemmer.stem(token) for token in cell]
    return stemmed_tokens


if __name__ == "__main__":
    result = predict(
        "<p>i would like to create a sql \
        script to seed the database when \
        my program first booooot 000 \
        <code>something that will be \
        removed</code></p>")
    print(result)

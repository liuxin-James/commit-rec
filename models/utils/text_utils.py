from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

# compute text similarity; return torch
def compute_text_similarity(sentence1: str, sentence2: str):
    sentence1 = sent_tokenize(preprocess_sentence(sentence1))
    sentence2 = sent_tokenize(preprocess_sentence(sentence2))

    embedding1 = model.encode(sentences=sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentences=sentence2, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embedding1, embedding2)

    return cosine_scores


# preprocess for sentence
def preprocess_sentence(sentence: str):
    before_words = word_tokenize(sentence)
    after_words = [
        word for word in before_words if word not in stopwords.words("english")]
    sentence_ = " ".join(after_words)
    return sentence_


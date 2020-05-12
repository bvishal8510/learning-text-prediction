import re
from collections import Counter

def preprocess(raw_text):

    # Replace punctuation with tokens so we can use them in our model

    reg = re.compile(r'\([a-z0-9 ,/]*\)')
    matches = re.findall(reg,raw_text)
    for match in matches:
        raw_text = raw_text.replace(match, '')
    reg = re.compile(r'\[[a-z0-9 ,/]*\]')
    matches = re.findall(reg,raw_text)
    for match in matches:
        raw_text = raw_text.replace(match, '')

    unique_char = list(set(raw_text))
    for i in unique_char:
        if(i=='.' or i=='?' or i=='!' or i=='\n'):
            if(i=='\n'):
                raw_text = raw_text.replace(i, ' ')
            else:
                raw_text = raw_text.replace(i, '.')
        elif((not i.isalnum()) and i!=' ' ):
            raw_text = raw_text.replace(i, '')

    sentences = raw_text.split('.')
    raw_text = raw_text.replace('.',' ')
    words = raw_text.split()

    # Remove all words with  5 or fewer occurences
    # word_counts = Counter(words)
    # trimmed_words = [word for word in words if word_counts[word] > 5]

    return words, sentences

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict....
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab
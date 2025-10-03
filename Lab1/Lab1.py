import nltk
import pymorphy3
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')

morph = pymorphy3.MorphAnalyzer()

with open("text.txt", encoding="utf-8") as f:
    text = f.read()

tokens = word_tokenize(text, language='russian')

def parse_word(word):
    parsed = morph.parse(word)
    if parsed:
        return parsed[0]
    return None

parsed_tokens = [parse_word(t) for t in tokens] #if t.isalpha()

def agree(w1,w2):
    if w1.tag.number != w2.tag.number:
        return False

    if w1.tag.gender is not None and w2.tag.gender is not None:
        if w1.tag.gender != w2.tag.gender:
            return False

    if w1.tag.case is not None and w2.tag.case is not None:
        if w1.tag.case != w2.tag.case :
            return False
    return True

for i in range(len(parsed_tokens)-1):
    w1, w2 = parsed_tokens[i], parsed_tokens[i+1]
    if not w1 or not w2:
        continue
    if not (w1.tag.POS in {"NOUN","ADJF"} and w2.tag.POS in {"NOUN","ADJF"}):
        continue

    if agree(w1,w2):
        print(w1.normal_form, w2.normal_form)

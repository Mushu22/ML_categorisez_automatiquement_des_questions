from flask import Flask, request, jsonify
import joblib
import os, sys

import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import __main__


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet')
    
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


workspace_directory = ""
app = Flask(__name__)


def text_tokeniser(text):
    
    def merge_csharp_bigram(tokens):
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "c" and i + 1 < len(tokens) and tokens[i + 1] == "#":
                merged_tokens.append("c#")
                # Sauter le token suivant
                i += 2  
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens
        
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return 'a'  # Adjectif
        elif treebank_tag.startswith('V'):
            return 'v'  # Verbe
        elif treebank_tag.startswith('N'):
            return 'n'  # Nom
        elif treebank_tag.startswith('R'):
            return 'r'  # Adverbe
        else:
            return 'n'  # Par défaut, utiliser nom
    
    # uncontract text
    expanded_text = contractions.fix(text)

    # Word Tokenizer
    tokens = word_tokenize(expanded_text)
    
    # POS Tagging
    pos_tags = pos_tag(tokens)

    # Lemmatisation avec POS
    lemmatized_tokens = []
    lemmatizer = WordNetLemmatizer()
    for token, tag in pos_tags:
        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(token, pos)
        lemmatized_tokens.append(lemma)

    # C# merging
    final_tokens = merge_csharp_bigram(lemmatized_tokens)
    
    return final_tokens


setattr(__main__, "text_tokeniser", text_tokeniser)

@app.route('/') 
def index():
    return "It works"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)

    pre_process = joblib.load(os.path.join(workspace_directory, "tfidf.joblib"))
    data_transformed = pre_process.transform([data])

    model = joblib.load(os.path.join(workspace_directory, "TFIDF_PassiveAgressive_model.joblib"))
    data_out = model.predict(data_transformed)
    print(data_out)

    mlb = joblib.load(os.path.join(workspace_directory, "mlb.joblib"))
    result = mlb.inverse_transform(data_out)

    print(result)
    return jsonify({'result': result})

if __name__ == '__main__':
    
    # workspace_directory = sys.argv[1]
    app.run(port=4125)


# from flask import Flask, request, render_template, jsonify
# from flask_cors import CORS
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# import json
# import random
# import numpy as np
# import pickle

# app = Flask(__name__)
# CORS(app)  # This will enable CORS for all routes

# # Load intents
# with open('intents.json') as file:
#     intents = json.load(file)

# # Load tokenizer
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)

# max_sequence_length = 50  # Adjust based on your sequence length during training

# # Load the model architecture
# with open('model_architecture.json', 'r') as json_file:
#     loaded_model_json = json_file.read()
#     model = tf.keras.models.model_from_json(loaded_model_json)

# # Load the model weights
# model.load_weights('model_weights.h5')

# # Compile the model (you need to compile after loading weights)
# model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

# # Assume `classes` is the list of class labels
# classes = [intent['tag'] for intent in intents['intents']]

# def process_model_output(output, intents_json):
#     # Interpret the model output
#     predicted_index = np.argmax(output)
#     predicted_tag = classes[predicted_index]

#     # Fetch a response based on the predicted class
#     for intent in intents_json['intents']:
#         if intent['tag'] == predicted_tag:
#             return random.choice(intent['responses'])

#     return "Sorry, I didn't understand that."

# @app.route('/')
# def index():
#     return render_template('test.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     try:
#         user_message = request.json['message']

#         # Tokenize the input message
#         sequences = tokenizer.texts_to_sequences([user_message])

#         # Pad the sequence to the maximum length
#         padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length)

#         # Predict
#         response = model.predict(padded_sequence)

#         # Post-process the prediction
#         chat_response = process_model_output(response[0], intents)

#         return jsonify({'response': chat_response})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import json
import numpy as np
import random
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import string

nltk.download("punkt")
nltk.download("wordnet")

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('test.html')

# Load intents
with open('intents.json') as file:
    intents = json.load(file)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the model architecture
with open('model_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    model = tf.keras.models.model_from_json(loaded_model_json)

# Load the model weights
model.load_weights('model_weights.h5')

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

# Load words and classes
with open('words.pkl', 'rb') as file:
    words = pickle.load(file)

with open('classes.pkl', 'rb') as file:
    classes = pickle.load(file)

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in string.punctuation]
    return tokens

def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

def predict_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = [labels[r[0]] for r in y_pred]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0] if intents_list else 'no_match'
    list_of_intents = intents_json["intents"]
    for intent in list_of_intents:
        if intent["tag"] == tag:
            result = random.choice(intent["responses"])
            break
    else:
        result = "I'm sorry, I didn't understand that."
    return result

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    intents_list = predict_class(message, words, classes)
    response = get_response(intents_list, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
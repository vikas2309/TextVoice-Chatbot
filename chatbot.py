import json
import nltk
import random
import pickle
import numpy as np
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer

# Initializing Lancaster Stemmer
stemmer = LancasterStemmer()

# Loading dataset
with open('dataset/dataset.json') as file:
    data = json.load(file)

# Load pickle file
with open('data.pickle', 'rb') as f:
    words, labels, train, output = pickle.load(f)

# Building the model with TensorFlow Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(train[0]),)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Loading the model
model.load_weights('models/chatbot-model.h5')

# Create a bag of zeros with length equal to the number of words in the vocabulary
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    # Tokenize the input sentence
    words = nltk.word_tokenize(s)
    # Stem and lowercase each token
    words = [stemmer.stem(word.lower()) for word in words]
    # Iterate through each word in the vocabulary
    for word in words:
        # Iterate through each word in the tokenized sentence
        for i, w in enumerate(words):
            # If the word in the vocabulary matches the word in the tokenized sentence, set the corresponding index in the bag to 1
            if w == word:
                bag[i] = 1

    # Convert the bag into a numpy array
    return np.array(bag)


def chat(inputText):
    inputText = inputText.lower()
    if inputText == 'quit' or inputText == 'bye' or inputText == 'Thank you':
        return "Goodbye!"

    # Predicting input sentence tag
    predict = model.predict(np.array([bag_of_words(inputText, words)]))
    predictions = np.argmax(predict)
    
    tag = labels[predictions]
    
    # Printing response
    for t in data['intents']:
        if t['tag'] == tag:
            responses = t['responses']
            
    outputText = random.choice(responses)
    return outputText


if __name__ == "__main__":
    print(chat("Hello"))
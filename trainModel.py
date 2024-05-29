import json  
import nltk  
import random  
import pickle  # Importing pickle for serializing and deserializing Python objects
import numpy as np  
import tensorflow as tf  
from nltk.stem.lancaster import LancasterStemmer  

# Initializing Lancaster Stemmer
stemmer = LancasterStemmer()

# Loading dataset from JSON file
with open('dataset/dataset.json') as f:
    data = json.load(f)

# Trying to load preprocessed data from pickle file
try:
    with open('data.pickle', 'rb') as file:
        words, labels, train, output = pickle.load(file)

# If no pickle file found, preprocess the data
except:
    words = []  # List to store all words in dataset
    x_docs = []  # List to store patterns (sentences)
    y_docs = []  # List to store tags for patterns
    labels = []  # List to store unique tags

    # Looping over all intents in the dataset
    for intent in data['intents']:
        # Looping over patterns (input sentences) in each intent
        for pattern in intent['patterns']:
            # Tokenizing each word in the pattern
            tokenizedWords = nltk.word_tokenize(pattern)
            # Extending words list with tokens
            words.extend(tokenizedWords)
            # Appending pattern and its tag to respective lists
            x_docs.append(tokenizedWords)
            y_docs.append(intent['tag'])
            # Appending unique tags to labels list
            if intent['tag'] not in labels:
                labels.append(intent['tag'])

    # Sorting labels list
    labels = sorted(labels)

    # Stemming words and sorting them
    words = [stemmer.stem(w.lower()) for w in words if w not in '?']
    words = sorted(list(set(words)))

    train = []  # List to store training data
    output = []  # List to store output data

    # Creating a Bag of Words - One Hot Encoding
    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(x_docs):
        bag = []
        stemmedWords = [stemmer.stem(w) for w in doc]

        # Marking word index as 1 if it exists in the pattern
        for w in words:
            if w in stemmedWords:
                bag.append(1)
            else:
                bag.append(0)

        # Creating output row with one hot encoding for each tag
        outputRow = out_empty[:]
        outputRow[labels.index(y_docs[x])] = 1

        train.append(bag)
        output.append(outputRow)

    # Converting data into numpy arrays
    train = np.array(train)
    output = np.array(output)

    # Saving preprocessed data into pickle file
    with open('data.pickle', 'wb') as f:
        pickle.dump((words, labels, train, output), f)

# Building the neural network model with TensorFlow Keras
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(train[0]),)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(len(output[0]), activation='softmax')
])

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('Training Model...')

# Training the model
model.fit(train, output, epochs=120, batch_size=8, verbose=1)

# Saving the trained model
model.save('models/chatbot-model.h5')

print('Model successfully trained')

def bag_of_words(s, words):
    """Converts input sentence into a bag of words representation.

    Args:
        s (str): Input sentence.
        words (list): List of words in the vocabulary.

    Returns:
        numpy.ndarray: Bag of words representation of the input sentence.
    """
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

def chat():
    """Function to chat with the trained chatbot."""
    print('Start talking...(type quit to exit)')
    while True:
        inp = input('You: ')

        # Type quit to exit
        if inp.lower() == 'quit':
            break

        # Predicting input sentence tag
        predict = model.predict(np.array([bag_of_words(inp, words)]))
        predictions = np.argmax(predict)
        
        tag = labels[predictions]
        # Printing response
        for t in data['intents']:
            if t['tag'] == tag:
                responses = t['responses']
                
        outputText = random.choice(responses)
        print(outputText)

# chat()

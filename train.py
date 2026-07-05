import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Training sentences
sentences = [
    "i love machine learning",
    "i love deep learning",
    "machine learning is interesting",
    "deep learning is powerful",
    "artificial intelligence is growing",
    "i want to learn ai",
    "ai can predict words",
    "rnn can learn sequences",
    "machine learning can predict results",
    "deep learning can generate sentences"
]

# 2. Convert words to numbers
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
total_words = len(word_index) + 1

print(word_index)
print("Total words:", total_words)

# 3. Create word-level training data
input_sequences = []

for sentence in sentences:
    token_list = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(token_list)):
        sequence = token_list[:i+1]
        input_sequences.append(sequence)

# 4. Make all sequences same length
max_sequence_len = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding="pre"
)

# 5. Split input and target
X_train = input_sequences[:, :-1]
y_train = input_sequences[:, -1]

# 6. Build RNN model
model = Sequential([
    Embedding(input_dim=total_words, output_dim=16),
    SimpleRNN(64),
    Dense(total_words, activation="softmax")
])

# 7. Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 8. Train model
model.fit(X_train, y_train, epochs=10, verbose=1)

# 9. Generate sentence
def generate_sentence(seed_text, next_words=5):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list],
            maxlen=max_sequence_len - 1,
            padding="pre"
        )

        prediction = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(prediction)

        predicted_word = ""

        for word, index in word_index.items():
            if index == predicted_index:
                predicted_word = word
                break

        seed_text += " " + predicted_word

    return seed_text

# 10. Test
print(generate_sentence("i love", next_words=4))
print(generate_sentence("machine learning", next_words=4))
print(generate_sentence("deep learning", next_words=4))
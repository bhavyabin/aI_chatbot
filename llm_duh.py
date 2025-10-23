
from dataset2 import data_text
from preprocess import text_cleaner
from sequences import create_seq
from sequence_encode import encode_seq
from train_test import split
from llm import generate_model
from load_llm import load
from inference import generate_seq



# preprocess the text
data_new = text_cleaner(data_text)

# create sequences
sequences = create_seq(data_new, length=30)

# create a character mapping index
chars = sorted(list(set(data_new)))
mapping = dict((c, i) for i, c in enumerate(chars))

# encode the sequences
encoded_sequences = encode_seq(sequences, mapping)

X_tr, X_val, y_tr, y_val, vocab = split(mapping, encoded_sequences)

model = generate_model(X_tr, X_val, y_tr, y_val, vocab)

if __name__ == "__main__":
    x = ""

    while x != "bye!":
        x = input(">>> ")

        print(generate_seq(model,mapping, 30, x.lower(), 50))

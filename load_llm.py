from keras.saving import load_model

def load():
    model = load_model("llm.h5")

    return model


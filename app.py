import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

@st.cache_resource
def load_model_and_tokenizer():
    """Loading the  trained model and tokenizer"""
    try:
        model = tf.keras.models.load_model("model_2.h5", compile=False)
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except FileNotFoundError:
        st.error("Model or tokenizer file not found.")
        return None, None

model, tokenizer = load_model_and_tokenizer()
max_sequence_len = 20

def generate_multiple_predictions(text, n_words=3, num_predictions=4):
    """ Generates 4 unique text predictions.
    """
    if model is None or tokenizer is None:
        return []

    predictions = []  # list to hold the prediction
    
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-num_predictions:][::-1]

    for start_index in top_indices:
        current_text = text
        result = []


        output_word = tokenizer.index_word.get(start_index, "")
        if not output_word:
            continue
        current_text += " " + output_word
        result.append(output_word)

        for _ in range(n_words - 1):
            token_list = tokenizer.texts_to_sequences([current_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted_index = np.argmax(predicted_probs)
            
            output_word = tokenizer.index_word.get(predicted_index, "")
            if not output_word:
                break
                
            current_text += " " + output_word
            result.append(output_word)
        
        predictions.append(" ".join(result))
        
    return predictions


st.title("Next Words Prediction")

st.markdown("""
Enter a starting text, choose how many words you want to predict, and click the button to see multiple generated sequences.
""")

input_text = st.text_area("Enter your text here:", height=100)
num_words = st.slider("Number of words to predict in each sequence:", min_value=1, max_value=20, value=5)

if st.button("Generate Predictions"):
    if input_text and model is not None:
        with st.spinner("Generating..."):
            predictions = generate_multiple_predictions(input_text.strip(), num_words, num_predictions=4)
            if predictions:
                st.success("Text prediction successfully generated!!")
                st.subheader("Generated Text Predictions:")
                for i, prediction in enumerate(predictions):
                    st.markdown(
                        f"""
                        <div style="background-color: #ababab; border-radius: 5px; padding: 15px; text-align: left; color: #333; margin-top: 10px; margin-bottom: 10px;">
                        {input_text.strip()} <strong style="color: #3d7dc1;">{prediction}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.warning("Could not generate any predictions. Please try a different input text.")
    elif not input_text:
        st.warning("Please enter some text to start the prediction.")


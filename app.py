import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # Required for BERT preprocessing

# Load the preprocessor and encoder using the Hub URLs
preprocessor_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"

preprocessor = hub.KerasLayer(preprocessor_url)
encoder = hub.KerasLayer(encoder_url)

# Example inference function
def get_bert_embeddings(text_input):
    inputs = preprocessor(tf.constant([text_input]))  # Preprocess text
    outputs = encoder(inputs)  # Get embeddings
    return outputs['pooled_output']  # Returns a tensor; convert to list for display

# Streamlit app
st.title("BERT Text Embedding Demo")

user_text = st.text_area("Enter text for embedding (e.g., a sentence or paragraph):", height=100)

if st.button("Generate Embeddings"):
    if user_text:
        with st.spinner("Processing with BERT..."):
            embeddings = get_bert_embeddings(user_text)
            # Convert tensor to list for nicer display (first 10 dims for brevity)
            embed_list = embeddings.numpy()[0][:10].tolist()
            st.write("Embeddings (first 10 dimensions shown):")
            st.json(embed_list)  # Or st.write(embed_list) for simple list
            st.info(f"Full embedding shape: {embeddings.shape}")
    else:
        st.warning("Please enter some text.")

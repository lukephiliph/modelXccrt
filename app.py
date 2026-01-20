import streamlit as st
from transformers import BertTokenizer, TFBertModel  # TF-based for compatibility
import tensorflow as tf
import numpy as np

# Load tokenizer and model (downloads ~420MB on first run)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Inference function
def get_bert_embeddings(text_input):
    inputs = tokenizer(text_input, return_tensors='tf', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.pooler_output  # Pooled embeddings (768 dims)

# Streamlit app
st.title("BERT Text Embedding Demo (Hugging Face)")

user_text = st.text_area("Enter text for embedding (e.g., a sentence or paragraph):", height=100)

if st.button("Generate Embeddings"):
    if user_text:
        with st.spinner("Processing with BERT..."):
            embeddings = get_bert_embeddings(user_text)
            # Convert to list (first 10 dims for display)
            embed_list = embeddings.numpy()[0][:10].tolist()
            st.write("Embeddings (first 10 dimensions shown):")
            st.json(embed_list)
            st.info(f"Full embedding shape: {embeddings.shape}")
    else:
        st.warning("Please enter some text.")

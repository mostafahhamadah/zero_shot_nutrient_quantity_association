import streamlit as st
st.title('Torch test')

try:
    import torch
    st.success(f'Torch {torch.__version__}')
except Exception as e:
    st.error(f'Torch failed: {e}')

try:
    from sentence_transformers import SentenceTransformer
    st.success('SentenceTransformer imported')
    model = SentenceTransformer('BAAI/bge-m3')
    st.success('Model loaded!')
    emb = model.encode(['hello'])
    st.success(f'Encoded shape: {emb.shape}')
except Exception as e:
    st.error(f'ST failed: {e}')

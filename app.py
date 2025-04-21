import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import string
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy
import torch
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

@st.cache_resource
def load_data():
    # Load data
    #st.write("Loading data and models... Please wait.")
    
    try:
        with open('contents/metadata_df.pkl', 'rb') as f:
            metadata_df = pickle.load(f)
        with open('contents/reviews_df.pkl', 'rb') as f:
            reviews_df = pickle.load(f)
        with open('contents/docs_df.pkl', 'rb') as f:
            docs_df = pickle.load(f)
        with open('contents/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('contents/tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        
        # Initialize model with proper device settings
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        
        return metadata_df, reviews_df, docs_df, vectorizer, tfidf_matrix, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None, None

# Define preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    
    pattern = r'[\[\]<>{}.,:;?\-!\"\'\`~@#$%^&*()_+=\\|/]'
    HTML_TAG_REGEX = re.compile(r'<[^>]+>')
    STOP_WORDS = set(stopwords.words('english'))
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001FA70-\U0001FAFF"  # Newer symbols
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002B50"            # Star
        "\U0000FE0F"            # Variation selector
        "]+",
        flags=re.UNICODE
    )
    
    text = HTML_TAG_REGEX.sub(' ', text)
    text = emoji_pattern.sub(' ', text)
    text = text.lower()
    text = re.sub(pattern, ' ', text)
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 1]
    text = ' '.join(filtered_words)
    
    return text


# Function to get image from URL
def get_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None

# Function to embed long documents
def embed_long_doc(text, max_tokens=512, stride=256, pool="mean", model=None, tokenizer=None):
    if not isinstance(text, str) or not text.strip():
        return torch.zeros(384)  # Return zero vector for empty text
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', use_fast=True)
    
    try:
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
        
        # If text is short enough, process it directly
        if len(tokens) <= max_tokens:
            with torch.no_grad():
                return model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        
        # For longer texts, use sliding window approach
        slices = []
        for i in range(0, len(tokens), stride):
            slice_tokens = tokens[i:i + max_tokens]
            slice_text = tokenizer.decode(slice_tokens)
            slices.append(slice_text)
        
        # Process slices in batches
        batch_size = 8  # Adjust based on your GPU memory
        slice_embeddings = []
        
        for i in range(0, len(slices), batch_size):
            batch = slices[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
                slice_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        all_embeddings = torch.cat(slice_embeddings, dim=0)
        
        if pool == "mean":
            return all_embeddings.mean(dim=0)
        elif pool == "max":
            return all_embeddings.max(dim=0).values
        else:
            raise ValueError("pool must be 'mean' or 'max'")
            
    except Exception as e:
        st.warning(f"Error in embedding document: {str(e)}")
        return torch.zeros(384)  # Return zero vector in case of error

# Recommendation functions
def recommend_products_based_on_tfidf(query, top_n=5):
    """Recommend products based on TF-IDF similarity."""
    # Get global variables
    docs_df = st.session_state.docs_df
    vectorizer = st.session_state.vectorizer
    tfidf_matrix = st.session_state.tfidf_matrix
    
    # Transform query
    q_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity
    cosine_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    
    # Get top indices
    top_indices = cosine_scores.argsort()[::-1][:50]
    
    # Create dataframe with candidate products
    candidate_products = docs_df.iloc[top_indices].copy()
    candidate_products['tfidf_score'] = cosine_scores[top_indices]
    
    # Sort by score and return top N
    candidate_products = candidate_products.sort_values('tfidf_score', ascending=False)
    candidate_products = candidate_products.reset_index(drop=True)
    
    return candidate_products.head(top_n)

def recommend_products_based_on_sentence_transformers(query, top_n=5):
    """Recommend products using a hybrid approach with TF-IDF and Sentence Transformers."""
    # Get tfidf candidates
    tfidf_result = recommend_products_based_on_tfidf(query, top_n=50)
    
    # Get model
    model = st.session_state.model
    
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Encode documents
    docs = tfidf_result['document'].tolist()
    doc_embeddings = torch.stack([embed_long_doc(text, model=model) for text in docs])
    
    # Calculate similarities
    sims = util.cos_sim(query_embedding, doc_embeddings).squeeze(0)
    
    # Add scores to dataframe
    tfidf_result['semantic_score'] = sims.cpu().tolist()
    
    # Calculate combined score
    tfidf_result['combined_score'] = (
        0.6 * tfidf_result['tfidf_score'] +
        0.4 * tfidf_result['semantic_score']
    )
    
    # Return top results
    return (tfidf_result
            .nlargest(top_n, 'combined_score')
            [['parent_asin', 'title_processed', 'combined_score']]
            .reset_index(drop=True))


# ----- Streamlit App -----

def main():
    # Set page title and icon
    st.set_page_config(
        page_title="Reviews based Recommender",
        layout="wide"
    )


    
    # Custom CSS to reduce padding at the top
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
            .stTitle {
                margin-top: -5rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Title and search method in the same row
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Reviews based Recommender")
    with col2:
        search_method = st.selectbox(
            "Select search method",
            ["TF-IDF based", "TF-IDF Hybrid based", "Semantic Search based"],
            index=1  # Default to TF-IDF Hybrid
        )
    
    # Add some space
    st.markdown("<br>", unsafe_allow_html=True)

    
    # Search bar (wider)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if 'search_text' not in st.session_state:
            st.session_state.search_text = ""
        prompt = st.text_input("Enter your product search", value=st.session_state.search_text, key="search_input", label_visibility="collapsed", placeholder="Search for products...")
        st.session_state.search_text = prompt
    
    # Buttons below search bar (centered)
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    with col3:
        reset_button = st.button("🔄 Reset", use_container_width=True)
    
    # Handle enter key press
    if prompt and not search_button and not reset_button:
        search_button = True
    
    # Initialize session state for history if it doesn't exist
    if 'history' not in st.session_state:
        st.session_state.history = []
    
            # Initialize data if not already done
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        with st.spinner("Loading data and models..."):
            metadata_df, reviews_df, docs_df, vectorizer, tfidf_matrix, model = load_data()
            
            # Store in session state
            st.session_state.metadata_df = metadata_df
            st.session_state.reviews_df = reviews_df
            st.session_state.docs_df = docs_df
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.model = model
            st.session_state.data_loaded = True
    
    # When search button is clicked
    if search_button and prompt:
        # Preprocess the prompt
        processed_prompt = preprocess_text(prompt)
        
        # Show loading spinner while processing
        with st.spinner("Searching for products..."):
            # Get recommendations based on selected method
            if search_method == "TF-IDF based":
                results = recommend_products_based_on_tfidf(processed_prompt)
                # Prepare results for display
                display_results = results[['parent_asin', 'title_processed', 'tfidf_score']].copy()
                display_results.columns = ['ASIN', 'Product Title', 'Score']
                
            elif search_method == "TF-IDF Hybrid based":
                results = recommend_products_based_on_sentence_transformers(processed_prompt)
                # Prepare results for display
                display_results = results.copy()
                display_results.columns = ['ASIN', 'Product Title', 'Score']
                
            else:  # Semantic Search based
                st.info("Semantic Search not implemented yet.")
                return
            
            # Add image column
            display_results['Image'] = None
            
            # Get image URLs for each product
            for i, row in display_results.iterrows():
                asin = row['ASIN']
                try:
                    # Find product in reviews_df
                    product_images = st.session_state.reviews_df[st.session_state.reviews_df['parent_asin'] == asin]['images'].values
                    
                    # Check if there are any images
                    if len(product_images) > 0:
                        # Get the first non-empty array
                        for img_array in product_images:
                            if len(img_array) > 0:
                                # Get the first image's large_image_url
                                img_url = img_array[0].get('large_image_url', None)
                                if img_url:
                                    img = get_image_from_url(img_url)
                                    if img:
                                        display_results.at[i, 'Image'] = img
                                        break
                except Exception as e:
                    continue
            
            # Store in history
            st.session_state.history.append({
                'prompt': prompt,
                'method': search_method,
                'results': display_results
            })
    
    # Reset search
    if reset_button:
        st.session_state.history = []
        st.session_state.search_text = ""
        st.rerun()
    
    # Display results if available
    if st.session_state.history and len(st.session_state.history) > 0:
        latest = st.session_state.history[-1]
        
        st.subheader(f"Search Results for: '{latest['prompt']}'")
        st.caption(f"Method: {latest['method']}")
        
        # Display results table
        results_df = latest['results'].copy()
        
        # Format the Score column to show fewer decimal places
        results_df['Score'] = results_df['Score'].apply(lambda x: f"{x:.4f}")
        
        # Display column headings
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            st.write("**ASIN**")
        with col2:
            st.write("**Product Title**")
        with col3:
            st.write("**Score**")
        with col4:
            st.write("**Image**")
        
        st.markdown("---")  # Separator after headings
        
        # Display table with images
        for i, row in results_df.iterrows():
            col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
            
            with col1:
                st.write(row['ASIN'])
            
            with col2:
                # Get the title from metadata_df using parent_asin
                product_title = st.session_state.metadata_df[
                    st.session_state.metadata_df['parent_asin'] == row['ASIN']
                ]['title'].values
                
                if len(product_title) > 0:
                    title = product_title[0]
                else:
                    title = row['Product Title']
                
                # Create Amazon search URL
                search_url = f"https://www.amazon.com/s?k={title.replace(' ', '+')}"
                st.markdown(f"[{title}]({search_url})", unsafe_allow_html=True)
            
            with col3:
                st.write(row['Score'])
            
            with col4:
                if row['Image'] is not None:
                    st.image(row['Image'], width=100)
                else:
                    st.write("No image available")
            
            st.markdown("---")  # Separator between rows

if __name__ == "__main__":
    main()

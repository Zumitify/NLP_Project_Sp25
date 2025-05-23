import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import torch
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
import pickle
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import asyncio
import os
nltk.download('stopwords')
nltk.download('punkt')
stemmer = PorterStemmer()

# Set the event loop policy
#asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

def download_from_github():
    # List of files to download
    files_to_download = [
        'embeddings_data.pkl',
        'vectorizer.pkl',
        'tfidf_matrix.pkl',
        'reviews_df.pkl',
        'metadata_df.pkl',
        'docs_df.pkl'
    ]
    
    # Base URL for raw GitHub content
    base_url = "https://raw.githubusercontent.com/Zumitify/NLP_Project_Sp25/main/contents/"

    if not os.path.exists('contents'):
        os.makedirs('contents')
    
    # Downloading each file if it doesn't exist
    for file in files_to_download:
        file_path = os.path.join('contents', file)
        if not os.path.exists(file_path):
            try:
                st.write(f"Downloading {file}...")
                url = base_url + file
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.write(f"Successfully downloaded {file}")
            except Exception as e:
                st.error(f"Error downloading {file}: {str(e)}")
                return False
    return True

if not download_from_github():
    st.error("Failed to download required files. Please check the logs for details.")
    st.stop()



def load_embeddings_and_model(filename='contents/embeddings_data.pkl'):
    try:
        # Loading data with explicit CPU mapping and pickle module
        data = torch.load(
            filename,
            map_location=torch.device('cpu'),
            pickle_module=pickle,
            weights_only=False
        )

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(data['model_name'], device=device)
        
        # Converting list embeddings to dictionary format
        metadata_embeddings_dict = {}
        review_embeddings_dict = {}
        
        for i, asin in enumerate(data['product_asins']):
            if i < len(data['metadata_embeddings']):
                metadata_embeddings_dict[asin] = torch.tensor(data['metadata_embeddings'][i])
            if i < len(data['review_embeddings']):
                review_embeddings_dict[asin] = torch.tensor(data['review_embeddings'][i])
        
        return (
            metadata_embeddings_dict,
            review_embeddings_dict,
            data['product_asins'],
            model
        )
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        return None, None, None, None


def find_similar_products(query, model, metadata_embeddings, product_asins, top_n=50):
    if metadata_embeddings is None or len(metadata_embeddings) == 0:
        st.warning("No product embeddings available.")
        return pd.DataFrame(), None
    
    if len(metadata_embeddings) != len(product_asins):
        st.error("Mismatch between number of embeddings and number of ASINs.")
        return pd.DataFrame(), None
    
    # Encode query
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Convert dictionary embeddings to tensor
    if isinstance(metadata_embeddings, dict):
        embeddings_list = []
        for asin in product_asins:
            if asin in metadata_embeddings:
                embeddings_list.append(metadata_embeddings[asin])
            else:
                embeddings_list.append(torch.zeros_like(next(iter(metadata_embeddings.values()))))

        metadata_embeddings_tensor = torch.stack(embeddings_list)
    else:
        metadata_embeddings_tensor = metadata_embeddings
    
    # Finding similar products
    hits = util.semantic_search(query_embedding, metadata_embeddings_tensor, top_k=top_n)
    
    # Creating initial results dataframe
    results = []
    if hits and hits[0]:
        for hit in hits[0]:
            asin = product_asins[hit['corpus_id']]
            score = hit['score']
            results.append({
                'parent_asin': asin,
                'similarity_score': score
            })
    
    # Converting to DataFrame and sorting by similarity_score in descending order
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('similarity_score', ascending=False)
    
    return results_df, query_embedding

def semantic_search_similar_products(similar_products_df, query_embedding, review_embeddings, metadata_embeddings, alpha=0.5):
    results = {}
    
    for idx, row in similar_products_df.iterrows():
        asin = row['parent_asin']
        try:
            # Getting the index of the ASIN in the embeddings
            asin_index = None
            
            if isinstance(metadata_embeddings, dict):
                if asin in metadata_embeddings:
                    asin_index = asin
                else:
                    continue
            elif isinstance(metadata_embeddings, torch.Tensor):
                asin_index = idx
            
            if asin_index is not None:
                # Get embeddings based on type
                if isinstance(metadata_embeddings, dict):
                    metadata_embedding = metadata_embeddings[asin]
                    review_embedding = review_embeddings[asin]
                else:
                    metadata_embedding = metadata_embeddings[asin_index]
                    review_embedding = review_embeddings[asin_index]
                
                # Ensuring embeddings are tensors
                if not isinstance(metadata_embedding, torch.Tensor):
                    metadata_embedding = torch.tensor(metadata_embedding)
                if not isinstance(review_embedding, torch.Tensor):
                    review_embedding = torch.tensor(review_embedding)

                combined_embedding = (alpha * metadata_embedding + (1 - alpha) * review_embedding)

                if len(query_embedding.shape) == 1:
                    query_embedding = query_embedding.unsqueeze(0)
                
                # Calculate similarity using semantic_search
                hits = util.semantic_search(query_embedding, [combined_embedding])

                if hits and hits[0]:
                    for hit in hits[0]:
                        score = hit['score']
                        results.setdefault(asin, score)
                else:
                    continue
                
        except Exception as e:
            continue
    
    return results

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
        
        # Try to load embeddings and model
        metadata_embeddings, review_embeddings, product_asins, model = load_embeddings_and_model()
        
        return metadata_df, reviews_df, docs_df, vectorizer, tfidf_matrix, model, metadata_embeddings, product_asins, review_embeddings
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None, None, None, None, None, None

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
    text = stemmer.stem(text)
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
        # Tokenizing the text
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max_tokens)
        
        # If text is short enough, process it directly
        if len(tokens) <= max_tokens:
            with torch.no_grad():
                return model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        
        # If text is longer than max_tokens, use sliding window approach
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
        
        # Combining all embeddings
        all_embeddings = torch.cat(slice_embeddings, dim=0)
        
        if pool == "mean":
            return all_embeddings.mean(dim=0)
        elif pool == "max":
            return all_embeddings.max(dim=0).values
        else:
            raise ValueError("pool must be 'mean' or 'max'")
            
    except Exception as e:
        st.warning(f"Error in embedding document: {str(e)}")
        return torch.zeros(384)

# Recommendation functions
def recommend_products_based_on_tfidf(query, top_n=5):
    # Get global variables
    docs_df = st.session_state.docs_df
    vectorizer = st.session_state.vectorizer
    tfidf_matrix = st.session_state.tfidf_matrix

    q_vec = vectorizer.transform([query])
    
    # Calculating cosine similarity
    cosine_scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    top_indices = cosine_scores.argsort()[::-1][:50]
    candidate_products = docs_df.iloc[top_indices].copy()
    candidate_products['tfidf_score'] = cosine_scores[top_indices]
    
    # Sort by score and return top N
    candidate_products = candidate_products.sort_values('tfidf_score', ascending=False)
    candidate_products = candidate_products.reset_index(drop=True)
    top_results = candidate_products.head(top_n)
    
    # Creating final results with proper titles from metadata_df
    results = pd.DataFrame({
        'parent_asin': top_results['parent_asin'],
        'title_processed': top_results['parent_asin'].apply(
            lambda x: st.session_state.metadata_df[
                st.session_state.metadata_df['parent_asin'] == x
            ]['title'].values[0] if len(st.session_state.metadata_df[
                st.session_state.metadata_df['parent_asin'] == x
            ]['title'].values) > 0 else 'Unknown'
        ),
        'tfidf_score': top_results['tfidf_score']
    })
    
    return results

def recommend_products_based_on_sentence_transformers(query, top_n=5):
    tfidf_result = recommend_products_based_on_tfidf(query, top_n=50)
    model = st.session_state.model

    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Encode documents
    docs = tfidf_result['title_processed'].tolist()
    doc_embeddings = torch.stack([embed_long_doc(text, model=model) for text in docs])
    
    # Calculate similarities
    sims = util.cos_sim(query_embedding, doc_embeddings).squeeze(0)
    tfidf_result['semantic_score'] = sims.cpu().tolist()
    
    # Calculating combined score
    tfidf_result['combined_score'] = (
        0.6 * tfidf_result['tfidf_score'] +
        0.4 * tfidf_result['semantic_score']
    )
    
    # Getting top results 
    top_results = tfidf_result.nlargest(top_n, 'combined_score')
    
    # Creating final results with proper titles from metadata_df
    results = pd.DataFrame({
        'parent_asin': top_results['parent_asin'],
        'title_processed': top_results['parent_asin'].apply(
            lambda x: st.session_state.metadata_df[
                st.session_state.metadata_df['parent_asin'] == x
            ]['title'].values[0] if len(st.session_state.metadata_df[
                st.session_state.metadata_df['parent_asin'] == x
            ]['title'].values) > 0 else 'Unknown'
        ),
        'combined_score': top_results['combined_score']
    })
    
    return results


###### Streamlit App

def main():
    st.set_page_config(
        page_title="Reviews based Recommender",
        layout="wide"
    )

    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
            .stTitle {
                margin-top: -3rem;
            }
        </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Reviews based Recommender")
    with col2:
        search_method = st.selectbox(
            "Select search method",
            ["TF-IDF based", "TF-IDF Hybrid based", "Semantic Search based"],
            index=0  # Default to TF-IDF method
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        if 'search_text' not in st.session_state:
            st.session_state.search_text = ""
        prompt = st.text_input("Enter your product search", value=st.session_state.search_text, key="search_input", label_visibility="collapsed", placeholder="Search for products...")
        st.session_state.search_text = prompt
    
    # Search bar buttons
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    with col3:
        reset_button = st.button("🔄 Reset", use_container_width=True)

    if prompt and not search_button and not reset_button:
        search_button = True

    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Initializing data if not already done
    if 'data_loaded' not in st.session_state or not st.session_state.data_loaded:
        with st.spinner("Loading data and models..."):
            try:
                metadata_df, reviews_df, docs_df, vectorizer, tfidf_matrix, model, metadata_embeddings, product_asins, review_embeddings = load_data()

                st.session_state.metadata_df = metadata_df
                st.session_state.reviews_df = reviews_df
                st.session_state.docs_df = docs_df
                st.session_state.vectorizer = vectorizer
                st.session_state.tfidf_matrix = tfidf_matrix
                st.session_state.model = model
                st.session_state.metadata_embeddings = metadata_embeddings
                st.session_state.product_asins = product_asins
                st.session_state.review_embeddings = review_embeddings
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
    
    # When search button is clicked
    if search_button and prompt:
        # Preprocess the prompt
        processed_prompt = preprocess_text(prompt)

        with st.spinner("Searching for products..."):
            # Get recommendations based on selected method
            if search_method == "TF-IDF based":
                results = recommend_products_based_on_tfidf(processed_prompt)
                display_results = results[['parent_asin', 'title_processed', 'tfidf_score']].copy()
                display_results.columns = ['ASIN', 'Product Title', 'Score']
                
            elif search_method == "TF-IDF Hybrid based":
                results = recommend_products_based_on_sentence_transformers(processed_prompt)
                display_results = results.copy()
                display_results.columns = ['ASIN', 'Product Title', 'Score']
                
            else:  # Semantic Search based
                if st.session_state.metadata_embeddings is None or st.session_state.product_asins is None:
                    st.error("Semantic search embeddings not available. Please ensure embeddings_data.pkl exists.")
                    return
                
                if st.session_state.review_embeddings is None:
                    st.error("Review embeddings not available. Please ensure review_embeddings.pkl exists.")
                    return
                
                # Getting initial results and query embedding
                initial_results, query_embedding = find_similar_products(
                    processed_prompt,
                    st.session_state.model,
                    st.session_state.metadata_embeddings,
                    st.session_state.product_asins
                )
                
                if initial_results.empty:
                    st.warning("No results found using semantic search.")
                    return
                
                # Performing semantic search on similar products
                semantic_results = semantic_search_similar_products(
                    initial_results,
                    query_embedding,
                    st.session_state.review_embeddings,
                    st.session_state.metadata_embeddings
                )
                
                # Getting top 5 results
                top_results = dict(sorted(semantic_results.items(), key=lambda x: x[1], reverse=True)[:5])
                
                # Creating display results
                display_results = pd.DataFrame({
                    'ASIN': list(top_results.keys()),
                    'Score': list(top_results.values())
                })
                
                # Adding product titles for Semantic Search
                display_results['Product Title'] = display_results['ASIN'].apply(
                    lambda x: st.session_state.metadata_df[
                        st.session_state.metadata_df['parent_asin'] == x
                    ]['title'].values[0] if len(st.session_state.metadata_df[
                        st.session_state.metadata_df['parent_asin'] == x
                    ]['title'].values) > 0 else 'Unknown'
                )
            
            # Adding image 
            display_results['Image'] = None
            
            # Getting image URLs for each product
            for i, row in display_results.iterrows():
                asin = row['ASIN']
                try:
                    product_images = st.session_state.reviews_df[st.session_state.reviews_df['parent_asin'] == asin]['images'].values
                    if len(product_images) > 0:
                        for img_array in product_images:
                            if len(img_array) > 0:
                                img_url = img_array[0].get('large_image_url', None)
                                if img_url:
                                    img = get_image_from_url(img_url)
                                    if img:
                                        display_results.at[i, 'Image'] = img
                                        break
                except Exception as e:
                    continue
            
            # Storing in history
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
    
    # Displaing results
    if st.session_state.history and len(st.session_state.history) > 0:
        latest = st.session_state.history[-1]
        
        st.subheader(f"Search Results for: '{latest['prompt']}'")
        st.caption(f"Method: {latest['method']}")

        results_df = latest['results'].copy()
        # Formating the Score column to show fewer decimal places
        results_df['Score'] = results_df['Score'].apply(lambda x: f"{x:.4f}")

        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        with col1:
            st.write("**ASIN**")
        with col2:
            st.write("**Product Title**")
        with col3:
            st.write("**Score**")
        with col4:
            st.write("**Image**")
        st.markdown("---")
        
        # Displaing table with images
        for i, row in results_df.iterrows():
            col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
            
            with col1:
                st.write(row['ASIN'])
            
            with col2:
                # Creating Amazon search URL
                search_url = f"https://www.amazon.com/s?k={row['Product Title'].replace(' ', '+')}"
                st.markdown(f"[{row['Product Title']}]({search_url})", unsafe_allow_html=True)
            
            with col3:
                st.write(row['Score'])
            
            with col4:
                if row['Image'] is not None:
                    st.image(row['Image'], width=100)
                else:
                    st.write("No image available")
            
            st.markdown("---")

if __name__ == "__main__":
    main()

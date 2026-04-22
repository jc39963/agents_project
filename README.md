# Find Your Fit: Agentic Fashion Stylist

This project creates two approaches for building an intelligent digital fashion stylist that analyzes a user's uploaded clothing item and recommends complementary pieces from a product catalog. 

To explore different methodologies, the project implements two distinct agent architectures: a **Deep Learning Agent** powered by multimodal LLMs and CLIP embeddings for perception and planning, and a **Non-Deep Learning Agent** powered by traditional computer vision (SVM, HOG features) and color theory.

---

## Features

### Deep Learning Agent (`src/dl/agent.py`)
- **LLM-Driven Routing:** Uses `gpt-4o-mini` with function calling to orchestrate the styling process.
- **Multimodal Understanding:** Analyzes the user's uploaded image to determine style, aesthetic, and potential matches.
- **Fashion-CLIP Embeddings:** Embeds text descriptions of generated clothing concepts to search for visual matches in the catalog.
- **Vector Search:** Utilizes Pinecone to find the top-$k$ most similar Zara catalog items based on cosine similarity.
- **Interactive UI:** Designed to integrate with a Streamlit frontend, logging "thought" processes and execution steps interactively.

### Non-Deep Learning Agent (`src/non_dl/agent.py`)
- **SVM Classification:** Uses an SVM model alongside Histogram of Oriented Gradients (HOG) feature extraction and PCA to categorize the inputted clothing item (e.g., shirt, pants, dress).
- **Algorithmic Color Matching:** Extracts the dominant RGB color of the item and uses a triadic color harmony algorithm to find matching color palettes.
- **Rule-Based Pairing:** Suggests clothing categories that pair well together based on predefined fashion rules (e.g., hats go with jackets and jeans).
- **Composite Scoring Search:** Queries Pinecone using both image feature vectors and color vectors, calculating a weighted composite score to return the best matches.

### 📊 Evaluation Suite (`src/eval.py`)
- **Robustness Testing:** Tests agent resilience by automatically applying Gaussian blur and darkening perturbations to input images and calculating the percentage overlap of the resulting recommendations against a baseline.
- **LLM-as-a-Judge Aesthetic Evaluation:** Uses `gpt-4o` as an expert fashion judge to score the final outfit recommendations out of 5 based on:
  - Color Harmony
  - Style Cohesion
  - Proportion and Silhouette

---

## 📂 Project Structure

```text
agents_project/
│
├── data/
│   ├── images/                 # Test images and user uploads
│   ├── zara_combined.csv       # Zara product catalog metadata
│   └── database.py             # Database connection setup
│
├── src/
│   ├── dl/
│   │   ├── agent.py            # GPT-4o-mini agent and Streamlit loop
        ├── generate_embeddings.py                      # populates `zara-images` Pinecone index with Zara images embeddings

│   │   └── matching.py         # DL tool schemas and CLIP embedding logic
│   │
│   ├── non_dl/
│   │   ├── agent.py            # OpenAI function-calling agent for Non-DL
│   │   ├── color_match.py      # Triadic color theory logic
│   │   ├── identify_color.py   # Dominant RGB extraction
│   │   ├── identify_type.py    # SVM/HOG image classification
│   │   ├── item_combos.py      # Rule-based clothing pairings
│   │   └── search_catalog.py   # Composite Pinecone vector search
│   │
│   ├── eval.py                 # Evaluation suite (robustness, aesthetic scoring)
│   └── utils.py                # Pinecone DB initialization and shared utilities
```

---

## 🚀 Setup & Installation

1. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

2. **Install Requirements:**  
    ```bash
    pip install -r requirements.txt
    ``` 

3. **Data Requirements:**
   Ensure the Zara dataset or your dataset of products is located at `data/zara_combined.csv` and contains standard columns (e.g., `reference`, `image_url`, `name`). You will also need to populate your Pinecone indexes (`zara-images`, `product-non-dl-features`, `product-non-dl-colors`) beforehand.

4. **Running the App:**  

    ```bash
    streamlit run app.py
    ```



5. **Running the Evaluation:**
   You can test both agents and evaluate their outputs by running the evaluation script:
   ```bash
   python -m src.eval
   ```

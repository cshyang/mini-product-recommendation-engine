# mini-product-recommendation-engine
This project uses a CSV file (mini-product-recommender-dataset) as a product catalog and a mini AI-based chatbot system will recommend products based on user queries.

### How does it work?
1. Clean the data in the CSV file and remove `null` product description field.
2. Tokenized the product description field using `nltk word_tokenize`.
3. Vectorized the product description field using `CountVectorizer`.
4. Vectorized the user query using `CountVectorizer`.
5. A simple  `cosine_similarity` is applied on the vectoried_query and vectoried_product_description to calculate the `similarity score`.
6. Return the items with the highest similarity score.
7. Deploy the function as FastAPI
8. User query through streamlit frontend to get the result.

<img width="779" alt="image" src="https://github.com/cshyang/mini-product-recommendation-engine/assets/45311586/abd5852f-4b31-42de-a0ea-f427fdb2f1e3">

## Startup 🚀
1. Create a virtual environment `conda create --name mini-rec `
2. Activate the environment  `conda activate llm-env`
3. Clone this repo `git clone https://github.com/cshyang/mini-product-recommender.git`
4. Go to the directory.
5. Install the required dependencies `pip install -r requirements.txt`
6. Add your OpenAI APIKey in .env file
7. Start FastAPI port   `python api.py`
8. Start the app `streamlit run app.py`
9. Start using the app!

## Stacks
1. nltk
2. sklearn
3. cosine_similarity
4. FastAPI
5. Streamlit
6. Duckdb
7. LangChain
8. OpenAI

## Contribution
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Contact
GitHub: @cshyang


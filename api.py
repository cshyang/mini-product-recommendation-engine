import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer
import duckdb
from fastapi import FastAPI, HTTPException

#laod csv
csv = 'mini-product-recommender-dataset.csv'
df = duckdb.read_csv(csv)

data = duckdb.sql("""
                        SELECT *, concat("Product Name", ' ', Description) as feature 
                        FROM df where Description is not null
                        """).df()

# Download NLTK tokenizer data
nltk.download('punkt')

# Tokenization of Product Descriptions
data['feature_tokens'] = data['feature'].apply(lambda x: word_tokenize(str(x)))

# Create a vectorizer to convert descriptions to vectors
vectorizer = CountVectorizer(stop_words='english')
description_vectors = vectorizer.fit_transform(data['feature_tokens'].apply(' '.join))

# Function to get recommendation based on input
def get_recommendations(query, num_recommendations = 3):
    # Tokenize the user's query
    query_tokens = word_tokenize(query)
    
    # Convert the query to a vector
    query_vector = vectorizer.transform([' '.join(query_tokens)])
    
    # Calculate cosine similarities between the query and product descriptions
    query_similarities = cosine_similarity(query_vector, description_vectors)
    
    # Sort the products by similarity and exclude the queried product
    similar_products=data
    similar_products['Similarity'] = query_similarities[0]
    similar_products = similar_products.sort_values(by='Similarity', ascending=False)
    
    # Return the top 'num_recommendations' recommended products
    top_recommended_products = similar_products.head(num_recommendations)[['Product Name','Similarity']]
    
    # reset the dataframe index
    similar_products = similar_products.reset_index(drop=True)
    
    return top_recommended_products.to_dict(orient='records')



########## API Route ############
app = FastAPI()

@app.post("/recommend/")
async def recommend_products(query: str):
    try:
        recommendations = get_recommendations(query)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while getting recommendations.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# user_query = "Something casual suitable for travel"
# top_recommendations = get_recommendations(user_query)
# print("Top 3 recommended products:")
# print(top_recommendations)

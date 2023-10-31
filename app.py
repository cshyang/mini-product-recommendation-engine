import os
import requests
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import streamlit as st
from dotenv import load_dotenv

def main():
    # Prompt template
    chat_template = PromptTemplate(
    input_variables=["recomended_items"],
    template="""
        You are a friendly professional sales agent, you will receive a list of 3 items from a mini product recommender
        based on the user's description of an item. Here is an example of the item list
        ["Product Name": "Smartphone A", "Similarity": 0.3779644730092272,
        "Product Name": "Laptop B", "Similarity": 0,
        "Product Name": "Board Game U", "Similarity": 0]
        if the items have a Similarity score greater than 0, it means the items matched the description of the user; the higher
        the score, the more similar the items to the user's description. 
        Your job is to recommend the item with score greater than zero to the user and try to promote
        items with a score of 0 to the user as they might like the items. 
        Be friendly, use percentage insteaad of decimal when talk about the score, talk like a normal human, and greet the user. Keep the conversation lean.
        Here are the results of the mini recommendation engine:
      {recomended_items}""",
)
    # create llm
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    llm_chain = LLMChain(llm=llm, prompt=chat_template)
    
    # streatlit UI
    st.title("🔗🏷️ Mini Product Recommender")
    user_input = st.text_input("Describe a product of your interest here.")
    user_query = user_input

    # get recommendation
    api_url = "http://localhost:8000/recommend/" 
    params = {'query':user_query}
    response = requests.post(api_url, params=params)
    if response.status_code == 200:
        recommendations = response
        print("Top Product Recommendations:")
        for i, recommendation in enumerate(recommendations):
            print(f"{i + 1}. {recommendation}")
        else:
            print(f"Request failed with status code: {response.status_code}")

    if recommendations:
        with get_openai_callback() as callback:
            response = llm_chain.run(recommendations)
            print(callback)

    st.write(response)

if __name__ == "__main__":
    main()

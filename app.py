from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain_groq import ChatGroq

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
Groq_key = os.environ.get('Groq_key')
HF_TOKEN=os.environ.get('HF_TOKEN')



embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
# pinecone.init(api_key=PINECONE_API_KEY,
#               environment=PINECONE_API_ENV)

# Pinecone client initialize

from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("mchatbot")

llm = ChatGroq(api_key=Groq_key,model="llama3-70b-8192",temperature=0.3)

def query_pinecone(query_text, top_k=2):
    query_embedding = embeddings.encode(query_text).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

class CustomPineconeRetriever(BaseRetriever):
    def get_relevant_documents(self, query: str) -> List[Document]:
        results = query_pinecone(query)
        docs = []
        for match in results['matches']:
            metadata = match['metadata']
            text = metadata.pop('text', '')  # Remove 'text' from metadata and use it as the main content
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

# Initialize the language model


# Set up a custom prompt template
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Answer:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}

# Create the custom retriever
custom_retriever = CustomPineconeRetriever()

# Set up the RetrievalQA chain with the custom retriever
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=custom_retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)



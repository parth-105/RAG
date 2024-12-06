from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
# from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

# Create Google Generative AI model
llm = ChatGoogleGenerativeAI(api_key="AIzaSyBu0POnbbEqqu_6MnhrGDitEcVKKQH2WYw", model="gemini-1.5-pro", temperature=0.1)
# Initialize Google Generative AI embeddings
instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt")
    data = loader.load()

    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)

    # Save vector database locally
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load the vector database from the local folder
    # vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings, allow_dangerous_deserialization=True)

    # Create a retriever for querying the vector database
   
    retriever = vectordb.as_retriever(score_threshold=0.7)

    system_prompt = ("Use the given context to answer the question. " "In the answer, try to provide as much text as possible from the \"response\" section in the source document context without making many changes. " "If the answer is not found in the context, kindly state \"I don't know.\" Don't try to make up an answer. " "Context: {context}")
    prompt = ChatPromptTemplate.from_messages( [ ("system", system_prompt), ("human", "{input}"), ] )
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)
    # docs = retriever.invoke("javascript course")
    # print(docs)

    

 

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""




    return chain

if __name__ == "__main__":
    chain = get_qa_chain() 
    query = "Do you have python course?" 
    response = chain.invoke({"input": query}) 
    print(query)
    print(response['answer'])


# from langchain_google_genai import ChatGoogleGenerativeAI

# def check_import():
#     try:
#         # Create an instance of the ChatGoogleGenerativeAI model
#         llm = ChatGoogleGenerativeAI(api_key="AIzaSyBu0POnbbEqqu_6MnhrGDitEcVKKQH2WYw", model="gemini", temperature=0.1)
#         print("Import successful and model instance created!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     check_import()


# from langchain_google_genai import ChatGoogleGenerativeAI

# def ask_question(question):
#     try:
#         # Create an instance of the ChatGoogleGenerativeAI model
#         llm = ChatGoogleGenerativeAI(api_key="AIzaSyBu0POnbbEqqu_6MnhrGDitEcVKKQH2WYw", model="gemini-pro", temperature=0.1)
#         result = llm.invoke("Write a ballad about LangChain")
#         print(result.content)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Ask some questions to the model
#     ask_question("What is the capital of France?")
  


# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# def check_embeddings(texts):
#     try:
#         # Initialize Google Generative AI embeddings
#         embeddings_model = GoogleGenerativeAIEmbeddings(api_key="AIzaSyBu0POnbbEqqu_6MnhrGDitEcVKKQH2WYw", model="models/embedding-001")
        
#         # Generate embeddings for the provided texts
#         embeddings = embeddings_model.embed_query("hello, world!")
#         embeddings[:5]
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Sample texts to generate embeddings
#     sample_texts = "hello, world!"
    
#     check_embeddings(sample_texts)


# from langchain_huggingface import HuggingFaceEmbeddings



# def ask_question(question):
#     try:
#         # Create an instance of the ChatGoogleGenerativeAI model
#         embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#         vector = embeddings.embed_query("Hello, world!")
#         print(vector)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     # Ask some questions to the model
#     ask_question("What is the capital of France?")

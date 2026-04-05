from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from openai import OpenAI
from dotenv import load_dotenv
import os

# model = init_chat_model("gpt-5.2")

load_dotenv()
# print(os.getenv("OPENAI_API_KEY"))
client = OpenAI()
file_path = "./nodejs_tutorial.pdf"


def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("Warning: No documents loaded from PDF")
            return
        
        # docs = []
        # for document in loader.lazy_load():
        #     # print(document)
        #     docs.append(document)
        #     break  

        return docs
    except FileNotFoundError:
        print(f"Error: PDF file not found at {file_path}")
    except IndexError:
        print("Error: Document index out of range")
    except Exception as e:
        print(f"Error loading PDF: {str(e)}")   


# TODO: unstructured pdf loader is not working properly, need to check the documentation and find out how to use it properly.
# def unStructured_pdf(file_path):
#     loader = UnstructuredPDFLoader(file_path, mode="elements")

#     docs = loader.load()
#     print(docs[1])  


def chunk_pdf(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    chunks = text_splitter.split_documents(document)
    return chunks


def vector_embeddings():
    try:
        embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        # dimensions=1024
        )
        # return embeddings.embed_documents(chunks)
        return embeddings
    except Exception as e:
        print(f"Error while creating embeddings : {str(e)}")

# def qdrant_vector_store(embeddings, docs):
#     url = "http://localhost:6333"
#     qdrant = QdrantVectorStore.from_documents(
#         docs,
#         embeddings,
#         url=url,
#         # prefer_grpc=True,
#         collection_name="nodejs_tuto",
#     ) 

#     print(f"Qdrant Vector Store created successfully at {url} with collection name 'nodejs_tuto'") 

#     return qdrant      


def qdrant_vector_store():
    collection_name="nodejs_tuto"
    client = QdrantClient(url="http://localhost:6333")
    embeddings = vector_embeddings()

    vector_size = len(embeddings.embed_query("sample text"))

    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        pdf = load_pdf(file_path)
        chunks = chunk_pdf(pdf)

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )

        document_ids = vector_store.add_documents(documents=chunks)

        return vector_store
    else:
        print(f"Collection '{collection_name}' already exists. Using existing collection.")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        return vector_store

    


# qdrant =qdrant_vector_store(embeddings, chunks)
vector_store = qdrant_vector_store()




# @tool(response_format="content_and_artifact")
# def retrieve_context(query: str):
#     """Retrieve information to help answer a query."""
#     retrieved_docs = vector_store.similarity_search(query, k=2)
#     serialized = "\n\n".join(
#         (f"Source: {doc.metadata.get('page', 'unknown')}\nContent: {doc.page_content}")
#         for doc in retrieved_docs
#     )
#     return serialized, retrieved_docs

query = input("Enter your question : ")


retrieved_docs = vector_store.similarity_search(query, k=2)
print("retrievved docs : ", retrieved_docs)
context = "\n\n".join(
    (f"Source: {doc.metadata}\nContent: {doc.page_content}")
    for doc in retrieved_docs
)

prompt = f"""
You are a helpful assistant.

Use the provided context to answer the question.
Include page number if available.

If the answer is not in the context, say "I don't know".

Context:
{context}
"""

# agent = create_agent(model, tools, system_prompt=prompt)


# query = (
#     "what is let, var, const in js?"
# )

# for event in agent.stream(
#     {"messages": [{"role": "user", "content": query}]},
#     stream_mode="values",
# ):
#     event["messages"][-1].pretty_print()


response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "low"},
    instructions=prompt,
    input=query,
)

print("\n" + response.output_text)
   

 




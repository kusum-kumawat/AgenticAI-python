from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
# client = OpenAI()
file_path = "./nodejs_tutorial.pdf"


def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            print("Warning: No documents loaded from PDF")
            return
        
        docs = []
        for document in loader.lazy_load():
            print(document)
            docs.append(document)

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, add_start_index=True,)
    chunks = text_splitter.split_documents(document)
    return chunks


def vector_embeddings(chunk):
    try:
        embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # With the `text-embedding-3` class
        # of models, you can specify the size
        # of the embeddings you want returned.
        # dimensions=1024
        )
        return embeddings.embed_documents(chunk)
    except Exception as e:
        print(f"Error while creating embeddings : {str(e)}")


def main():
    try:
        pdf = load_pdf(file_path)
        if not pdf:
            print("Please Provide pdf.")
            return
        chunks = chunk_pdf(pdf)
        print(len(chunks))

        for chunk in chunks:
            embeddings = vector_embeddings([chunk])

    except Exception as e:
        print(f"Error Occured : {str(e)}")

main()    




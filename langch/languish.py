
# [1]
import sys
# import document loader from lamgcahin
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma,Pinecone
from langchain import VectorDBQA
from langchain.llms import Cohere,OpenAI
from langchain.embeddings import CohereEmbeddings
import pinecone


# defining the cohere keys
COHERE_API_KEY = ''
EMBEDDING_TYPE = 'cohere'

#defining the loader
def load_data(data_path='https://django.readthedocs.io/_/downloads/en/latest/pdf/',loader_type='online'
              ):
    if loader_type == 'local':
        loader = UnstructuredPDFLoader(data_path)
    elif loader_type == 'online':
        loader = OnlinePDFLoader('https://django.readthedocs.io/_/downloads/en/latest/pdf/')
    data = loader.load()
    print(f"--- No of pages: {len(data)} \n")

    # Chunking up the data
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(data)
    print(f"--- No of chunks: {len(texts)}")
    return texts


def generate_embeddings(texts, embedding_type=EMBEDDING_TYPE):
    if embedding_type == 'cohere':
        embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    elif embedding_type == 'openai':
        embeddings = OpenAIEmbeddings(openai_api_key='OPENAI_API_KEY')

    return embeddings

def initialize_vecstore(embeddings,texts, vector_store='pinecone'):
    if vector_store == 'pinecone':
        #initialize pinecone
        pinecone.init( api_key='',environment='')
        index_name = 'cohere-docs-index'

        if index_name not in pinecone.list_indexes():
            if EMBEDDING_TYPE == 'cohere':
                pinecone.create_index(index_name, vector_size=4096, metric='cosine')
            elif EMBEDDING_TYPE == 'openai':
                pinecone.create_index(index_name, vector_size=768, metric='cosine')
    
        search_docs = Pinecone.from_texts([t.page_content for t in texts],embeddings, index_name=index_name)

    return search_docs

def initialize_llm(llmtype='cohere'):
    if llmtype == 'cohere':
        llm = Cohere(cohere_api_key=COHERE_API_KEY)
    elif llmtype == 'openai':
        llm = OpenAI(openai_api_key='OPENAI_API_KEY')
    return llm



def query_vecstore(search_docs, query,llm):
    topk_docs = search_docs.similarity_search(query, include_metadata=True)

    llm = initialize_llm()
    # qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=search_docs)
    chain = load_qa_chain(llm, chain_type="stuff")

    chain.run(query, topk_docs)

'''
search_docs = Pinecone.from_texts([t.page_content for t in texts],cohere_embeddings, index_name=index_name)
q = input("Enter your query: ")
print(search_docs.similarity_search(q,include_metadata=True))

# docsearch = Chroma.from_documents(texts, embeddings)

from langchain.llms import Cohere
cohere = Cohere(model="gptd-instruct-tft", cohere_api_key="lgi7A2ZBRIswmmUy3FIB0AbjfNhEnvWtgEXnElPi")

qa = VectorDBQA.from_chain_type(llm=cohere, chain_type="stuff", vectorstore=search_docs)
'''
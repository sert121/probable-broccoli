from langchain.vectorstores import Qdrant
import pinecone
from langchain.vectorstores import Chroma
from langchain.embeddings import CohereEmbeddings
import os
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import pickle
import cohere
from qdrant_client.http.models import Batch
from qdrant_client.http import models

COHERE_API_KEY = os.environ['COHERE_API_KEY']
COHERE_API_KEY = 'lgi7A2ZBRIswmmUy3FIB0AbjfNhEnvWtgEXnElPi'
cohere_client = cohere.Client(api_key=COHERE_API_KEY)

API_KEY_QDRANT = os.environ['QDRANT_KEY']

embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)


#initializing qdrant client cloud  ~~
def initialize_vecstore():
  client_q = QdrantClient(
    url=
    'https://5bcda451-5eec-489e-a663-1349d8693bf3.us-east-1-0.aws.cloud.qdrant.io:6333',
    api_key=API_KEY_QDRANT)

  collection_name = "Pandora"
  # client.create_collection(collection_name=collection_name,
  #                          vectors_config=VectorParams(
  #                            size=4096, distance=Distance.COSINE))

  qdrant = Qdrant(client_q,
                  collection_name,
                  embedding_function=embeddings.embed_documents)

  with open('gpt4.pkl', 'rb') as f:
    texts = pickle.load(f)

  fine_texts = [t.page_content for t in texts]
  # qdrant.from_documents(documents=texts,
  #                       embedding=embeddings,
  #                       collection_name=collection_name)
  # qdrant.add_texts(fine_texts)

  ids = [i for i in range(len(fine_texts))]
  embedded_vectors = cohere_client.embed(model="large",
                                         texts=fine_texts).embeddings
  # Conversion to float is required for Qdrant
  vectors = [list(map(float, vector)) for vector in embedded_vectors]

  client_q.upsert(collection_name=collection_name,
                  points=Batch(ids=ids, vectors=vectors))


def create_collection(collection_name='Pandora'):

  client_q = QdrantClient(
    url=
    'https://5bcda451-5eec-489e-a663-1349d8693bf3.us-east-1-0.aws.cloud.qdrant.io:6333',
    api_key=API_KEY_QDRANT)

  client_q.recreate_collection(
    collection_name=f"{collection_name}",
    vectors_config=models.VectorParams(size=4096,
                                       distance=models.Distance.COSINE),
  )


def get_collection():
  print("in here guys")
  client_q = QdrantClient(
    url=
    'https://5bcda451-5eec-489e-a663-1349d8693bf3.us-east-1-0.aws.cloud.qdrant.io:6333',
    api_key=API_KEY_QDRANT)
  collection_name = 'Pandora'
  details = client_q.get_collection(collection_name=f"{collection_name}")
  print(f"Details : {details}")


def query_vecstore(collection_name='Pandora', questions=['']):
  client_q = QdrantClient(
    url=
    'https://5bcda451-5eec-489e-a663-1349d8693bf3.us-east-1-0.aws.cloud.qdrant.io:6333',
    api_key=API_KEY_QDRANT)

  embedded_vectors = cohere_client.embed(model="large",
                                         texts=questions).embeddings
  # Conversion to float is required for Qdrant
  vectors = [list(map(float, vector)) for vector in embedded_vectors]
  k_max = 5

  response = client_q.search(collection_name=f"{collection_name}",
                             query_vector=vectors[0],
                             limit=k_max,
                             with_vectors=True,
                             with_payload=True)

  print(f'Response h: -----\n {response} \n-----')


def doc_store_lang():

  with open('gpt4.pkl', 'rb') as f:
    texts = pickle.load(f)
  host = 'https://5bcda451-5eec-489e-a663-1349d8693bf3.us-east-1-0.aws.cloud.qdrant.io:6333'
  fine_texts = [t.page_content for t in texts]
  doc_store = Qdrant.from_texts(fine_texts,
                                embeddings,
                                collection_name='ronan',
                                url=host,
                                api_key=API_KEY_QDRANT)
  print(doc_store)


# create_collection('ronan')
# get_collection()
# initialize_vecstore()
# query_vecstore(questions=['What is the size of gpt4?'])
doc_store_lang()

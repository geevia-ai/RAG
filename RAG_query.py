from RAG_save_document_vectors import VECTOR_FILE,model
import numpy as np
import json
import os
from openai import OpenAI
from numpy.typing import NDArray

VECTOR_RECORDS=[]
API_KEY= os.environ.get("API_KEY")
model_LLM = OpenAI(api_key=API_KEY)

def query_encode(query:str) -> NDArray:
    """
    create embedding for the query
    """
    embeddings = model.encode(
    query,
    normalize_embeddings=True
    )
    return embeddings

def create_document_vectors()-> None:
    """
    load embeddings from the vector store
    """
    with open(VECTOR_FILE,"r") as file:
        for record in file:
            VECTOR_RECORDS.append(json.loads(record))
    vectors = np.array(
        [record["embedding"] for record in VECTOR_RECORDS],
        dtype = np.float32
    )
    return vectors

def get_k_best_vectors(vectors: NDArray,query_vector: str,k:int =3) -> tuple[NDArray,list[float]]:
    """
    match the query with the vector store and return k highest matches
    """
    scores = vectors @ query_vector
    k_matches = np.argsort(scores)[::-1][:k]
    k_scores=[]
    for match in k_matches:
        k_scores.append(scores[match])
    return k_matches,k_scores

def model_query(k_matches:NDArray,k_scores:list[float],input_query:str) -> str:
    context_docs=[]
    for match_id,score in zip(k_matches,k_scores):
        vector_record = VECTOR_RECORDS[match_id]
        context_doc = { "chunk_id":vector_record["chunk_id"],
                         "source":vector_record["source"],
                         "text":vector_record["text"]
                      }
        context_docs.append(context_doc)
        print(f"score: {score}")
        print(context_doc)
    
    response = model_LLM.responses.create(
        model = "gpt-6-luna",
        instructions="""
        You answer questions only using the supplied documents(context_docs).
        Rules:
            - Answer only with information provided in the documents
            - The context docs have the following fields : chunk_id - the chunk referenced,source - the document referenced,text- the context
            - Cite factual information using the chunk_id,source fields in the document(context_doc), for instance  "chunk_id": "_index#chunk-1", "source": "docs/_index.md"
            - If the documents(context_docs) do not contain enough information to answer, replay exactly- I don't know based on the provided  sources
        """,
        input=f"""
        QUESTION : {input_query}

        DOCUMENTS :{context_docs}

        """
    )  

    return response.output_text

if __name__== "__main__":
    input_query = input("Enter query:")
    vector_query = query_encode(input_query)
    #print(vector_query)
    document_vectors = create_document_vectors()
    k_matches,k_scores = get_k_best_vectors(document_vectors,vector_query)
    response = model_query(k_matches,k_scores,input_query)
    print()
    print(f"Model response :{response}")
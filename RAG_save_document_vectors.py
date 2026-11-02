from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import json
from numpy.typing import NDArray
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
MAX_TOKENS_SIZE = model.max_seq_length

OVERLAP_SIZE = 50
KNOWLEDGE_VECTOR=[]
DOCUMENT_FILES = "./docs/"
VECTOR_FILE = "./vectors/vectors.jsonl"

def encode_text(text:str, doc_id:str) -> tuple[NDArray,list[str]]:
    """
    The model all-MiniLM-L6-v2 has a limitation of MAX_TOKENS_SIZE of 256 on the input text it can encode at a time.
    To get over this limitation, the text is first tokenized into MAX_TOKENS_SIZE size chunks
    Then the start and end positions are calculated for each token chunk using the offset_mapping and used to create text chunks
    The text chunks are toneknized again to ensure the MAX_TOKENS_SIZE  is not exceeded
    Finally the embeddings are created from the text chunks
    """

    if not model.tokenizer.is_fast:
        raise ValueError("Offset mapping requires a fast tokenizer.")
    
    encoded = model.tokenizer(
        text ,
        truncation = True,
        add_special_tokens=True,
        max_length = MAX_TOKENS_SIZE,
        stride = OVERLAP_SIZE,
        return_overflowing_tokens=True,
        return_offsets_mapping= True
    )


    chunks_text = []
    for offsets in encoded["offset_mapping"]:
        valid_offsets =[
            (start,end) for start,end in offsets
            if start < end 
        ]
        if not valid_offsets:
            continue
        start_position=valid_offsets[0][0]
        end_position = valid_offsets[-1][1]
        chunk = text[start_position:end_position]
        chunks_text.append(chunk)


    chunks_text_token_len = [
       len( 
            model.tokenizer(
            chunk ,
            add_special_tokens=True,
            truncation = False
            )['input_ids']
        ) for chunk in chunks_text
    ]

    chunks_text_token = [
       
            model.tokenizer(
            chunk ,
            add_special_tokens=True,
            truncation = False
            )['input_ids']
         for chunk in chunks_text
    ]

    oversized = [
        (index,count) for (index,count) in enumerate(chunks_text_token_len)
        if count > MAX_TOKENS_SIZE
    ]
    if oversized:
        for (id,_) in oversized:
                print(f"chunk text oversize:{chunks_text[id]}")
                print(f"chunk tokens:{chunks_text_token[id]}")
                print(f"original tokens:{encoded["input_ids"][id]}")
                print(f"end test")
        raise ValueError(f"document id:{doc_id} has the following chunks over {MAX_TOKENS_SIZE} : {oversized} ")
        
    embeddings = model.encode(
        chunks_text,
        normalize_embeddings=True
    )

    return embeddings,chunks_text

def content_cleanup(text: str)-> str:
    """
    cleans up wrapper,URLs from text
    """
    # Preserve glossary text
    text = re.sub(
        r'\{\{<\s*glossary_tooltip\b[^>]*\btext="([^"]*)"[^>]*>\}\}',
        r'\1',
        text
    )

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove Hugo shortcode wrappers
    text = re.sub(
        r'\{\{<\s*/?\s*[\w-]+(?:\s+[^>]*)?\s*>\}\}',
        '',
        text
    )

    # Preserve Markdown link text
    text = re.sub(r'!?\[([^\]]+)\]\([^)]+\)', r'\1', text)
    return text

def chunk_and_encode_file(doc_id: str,doc_path: str,content: str, vector_file: Path)-> None:
    """
    create documents with the cleaned document source, document id, chunk text and embeddings
    """
    content = content_cleanup(content)
    if not content.strip():
        return
    embeddings,chunks_text = encode_text(content,doc_id)
    for index,(embedding,chunk_text) in enumerate(zip(embeddings,chunks_text)):
        document = {
                        "doc_id":doc_id,
                        "chunk_id":f"{doc_id}#chunk-{index + 1}",
                        "source": doc_path,
                        "text": chunk_text,
                        "embedding": embedding.tolist()
                    }
        store_vector(document, vector_file)

def parse_save_documents(source_docs:Path,vector_file: Path, doc_ext:str='md') -> None:
    """
    Reads source documents to parse them
    """
    if not source_docs.is_dir():
        raise FileNotFoundError(f"Cannot find {source_docs} folder. Check the folder path")
    vector_file.parent.mkdir(parents=True, exist_ok=True)
    vector_file.write_text("",encoding="utf-8")

    documents = sorted(source_docs.rglob(f"*.{doc_ext}"))
    for doc_path in documents:
        if doc_path.is_file():
            doc_content = doc_path.read_text(encoding="utf-8")
            doc_id = Path(doc_path).relative_to(Path(DOCUMENT_FILES)).with_suffix("").as_posix()
            chunk_and_encode_file(doc_id,str(doc_path),doc_content,vector_file)
        else:
            raise FileNotFoundError(f"Cannot find {source_docs} folder. Check the folder path")

def store_vector(document: dict, vector_file:Path) -> None:
    """
    saves the created embedded document in a jsonl file
    """
    if not document:
        raise ValueError(f"{document} is empty")
   
    with open(str(vector_file),"a",encoding="utf-8") as file:
        file.write(json.dumps(document)+ "\n")

            

if __name__ == "__main__":
    source_docs = Path(DOCUMENT_FILES)
    vector_file = Path(VECTOR_FILE)
    parse_save_documents(source_docs,vector_file)
    print(f"Document vector sucessfully created at {VECTOR_FILE}")
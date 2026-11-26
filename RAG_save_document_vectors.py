from pathlib import Path
import re
from sentence_transformers import SentenceTransformer
import json
from numpy.typing import NDArray
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
OVERLAP_SIZE = 50
MAX_TOKENS_SIZE = model.max_seq_length
SPECIAL_TOKEN_SIZE = model.tokenizer.num_special_tokens_to_add(pair=False)
MAX_CONTENT_TOKENS = MAX_TOKENS_SIZE - SPECIAL_TOKEN_SIZE 


KNOWLEDGE_VECTOR=[]
DOCUMENT_FILES = "./docs/"
VECTOR_FILE = "./vectors/vectors.jsonl"

def encode_text(text:str, doc_id:str) -> tuple[NDArray,list[str]]:
    """
    The model all-MiniLM-L6-v2 has a limitation of MAX_TOKENS_SIZE of 256 on the input text it can encode at a time.
    To get over this limitation, the text is first tokenized as whole
    Then tokens that belong to the same word and grouped together and chunks with complete word tokens are created. Each chunk is capped at the (MAX_TOKENS_SIZE - SPECIAL_TOKEN_SIZE).
    Finally the embeddings are created from the text chunks
    """

    if not model.tokenizer.is_fast:
        raise ValueError("Offset mapping requires a fast tokenizer.")
    
    
    encoded = model.tokenizer(
        text,
        add_special_tokens= False,
        truncation=False,
        return_offsets_mapping= True
    )

    word_ids = encoded.word_ids()
    offsets = encoded["offset_mapping"]

    #group tokens that belong to the same word
    start_word_group = 0
    word_groups =[]
    while start_word_group < len(word_ids):
        end_word_group = start_word_group + 1
        current_word = word_ids[start_word_group]
        while(end_word_group < len(word_ids) and word_ids[end_word_group] == current_word):
            end_word_group += 1
        word_group = {
            "token_count":end_word_group - start_word_group,
            "char_start":offsets[start_word_group][0],
            "char_end":offsets[end_word_group - 1][1]
        }

        word_groups.append(word_group)
        start_word_group = end_word_group

    # now we have token start and end and counts for the same  word in the same word group.
    # next we chunk these tokens ,keeping all tokens for a word in the same chunk
    start_chunk_group = 0
    chunk_text_groups =[]
    chunk_length_groups =[]
    while start_chunk_group < len(word_groups):
        end_chunk_group = start_chunk_group
        current_chunk_count = 0
        while( end_chunk_group < len(word_groups) and current_chunk_count + word_groups[end_chunk_group]["token_count"] <= MAX_CONTENT_TOKENS):
            current_chunk_count += word_groups[end_chunk_group]["token_count"]
            end_chunk_group += 1

        if end_chunk_group == start_chunk_group:
            raise ValueError(f"chunk group{start_chunk_group} has tokens over the {MAX_CONTENT_TOKENS}")
        
        chunk_start_position = word_groups[start_chunk_group]["char_start"]
        chunk_end_position = word_groups[end_chunk_group - 1]["char_end"]
        chunk = text[chunk_start_position:chunk_end_position]
        chunk_text_groups.append(chunk)
        chunk_length_groups.append(current_chunk_count)
        if end_chunk_group == len(word_groups):
            break

        # calculate the next chunk start based on overlap words- find out what word to overlap starting from the last word of the current chunk and working backwords
        next_chunk_group = end_chunk_group
        overlap_tokens = 0
        while next_chunk_group -1 > start_chunk_group:
            previous_word_tokens = word_groups[next_chunk_group-1]["token_count"]
            if (overlap_tokens + previous_word_tokens > OVERLAP_SIZE):
                break
            next_chunk_group -=1
            overlap_tokens += previous_word_tokens

        start_chunk_group = next_chunk_group

    """
    chunks_text_token_len = [
       len( 
            model.tokenizer(
            chunk ,
            add_special_tokens=True,
            truncation = False
            )['input_ids']
        ) for chunk in chunk_text_groups
    ]
    """

    chunks_retokenized = [ 
            model.tokenizer(
            chunk ,
            add_special_tokens=True,
            truncation = False
            )['input_ids']
         for chunk in chunk_text_groups
    ]

    chunks_retokenized_len = [
        len(ids) for ids in chunks_retokenized
    ]

    oversized = [
        (index,(retoken_count,token_count)) for index,(retoken_count,token_count) in enumerate(zip(chunks_retokenized_len,chunk_length_groups))
        if retoken_count != token_count + SPECIAL_TOKEN_SIZE
    ]
    if oversized:
        for (id,(retoken_count,token_count)) in oversized:
                print(f"chunk text oversize:{chunk_text_groups[id]}")
                print(f"token_count:{token_count}")
                print(f"retoken_count:{retoken_count}")
                print(f"end test")
                break
        raise ValueError(f"document id:{doc_id} has the following chunks over {MAX_CONTENT_TOKENS} : {oversized} ")
        
    embeddings = model.encode(
        chunk_text_groups,
        normalize_embeddings=True
    )

    return embeddings,chunk_text_groups

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
    Parse source documents and save as vectors
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

import os
from PyPDF2 import PdfReader
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import cassio

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
ASTRA_DB_ID = "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

pdf_path = "speeches/budget_speech.pdf"
raw_text = ""
reader = PdfReader(pdf_path)
for page in reader.pages:
    content = page.extract_text()
    if content:
        raw_text += content

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="pdf_qa_ai_demo",
    session=None,
    keyspace=None,
)

astra_vector_store.add_texts(texts)
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

print("\nPDF-QA AI ready. Type 'quit' to exit.")

first_question = True
while True:
    prompt = (
        "\nEnter your question (or type 'quit' to exit): "
        if first_question
        else "\nYour next question (or type 'quit' to exit): "
    )
    query_text = input(prompt).strip()
    if query_text.lower() == "quit":
        break
    if not query_text:
        continue

    first_question = False
    print("\nQUESTION:", query_text)
    answer = astra_vector_index.query(query_text, llm=llm).strip()
    print("\nANSWER:", answer)

    print("\nTop relevant document snippets:")
    results = astra_vector_store.similarity_search_with_score(query_text, k=4)
    for i, (doc, score) in enumerate(results, start=1):
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f" {i}. [{score:.4f}] {snippet} ...")

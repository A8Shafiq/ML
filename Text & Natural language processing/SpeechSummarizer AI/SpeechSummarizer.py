import os
from PyPDF2 import PdfReader
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# ✅ SET YOUR OPENAI API KEY
openai_api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
os.environ["OPENAI_API_KEY"] = openai_api_key

# ✅ Initialize the ChatOpenAI instance
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# ✅ EXAMPLE speech text for summarization
speech_text = """
My dear friends, brothers, and sisters,

Today, as I stand before you, I do not stand merely as your representative in Parliament but as your companion...
[PASTE YOUR FULL SPEECH HERE IF YOU LIKE]
"""

# === STEP 1: Concise summary of the raw speech text ===
chat_messages = [
    SystemMessage(content="You are an expert assistant with expertise in summarizing speeches."),
    HumanMessage(content=f"Please provide a short and concise summary of the following speech:\n\n{speech_text}")
]

print("\n📝 Generating concise summary of the speech text...")
short_summary = llm(chat_messages).content
print("\n✅ Concise Summary:\n", short_summary)

# === STEP 2: Translate the summary to Hindi ===
translation_template = """
Write a summary of the following speech:
Speech: `{speech}`
Translate the precise summary to {language}.
"""

translation_prompt = PromptTemplate(
    input_variables=["speech", "language"],
    template=translation_template
)

translation_chain = LLMChain(llm=llm, prompt=translation_prompt)

print("\n🌐 Translating summary to Hindi...")
translated_summary = translation_chain.run({"speech": speech_text, "language": "Hindi"})
print("\n✅ Hindi Translation of Summary:\n", translated_summary)

# === STEP 3: Read text from a PDF file ===
pdf_file_path = "speeches/sample_speech.pdf"  # Change this path to your PDF

print(f"\n📂 Reading PDF: {pdf_file_path}...")
pdf_text = ""
try:
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            pdf_text += content
except Exception as e:
    print(f"❌ Error reading PDF: {e}")

print("\n✅ Extracted PDF Text (first 500 characters):\n", pdf_text[:500], "...")

# === STEP 4: Summarize the entire PDF text as a single chunk ===
doc = [Document(page_content=pdf_text)]

single_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Write a concise and short summary of the following speech:\nSpeech: `{text}`"
)

single_summary_chain = load_summarize_chain(
    llm=llm,
    chain_type="stuff",
    prompt=single_summary_prompt,
    verbose=False
)

print("\n📝 Generating single-chunk summary for the PDF...")
single_chunk_summary = single_summary_chain.run(doc)
print("\n✅ Single-Chunk PDF Summary:\n", single_chunk_summary)

# === STEP 5: Split large text into chunks and summarize with Map-Reduce ===
print("\n✂️ Splitting PDF text into manageable chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=20)
text_chunks = text_splitter.create_documents([pdf_text])

print(f"✅ Total chunks created: {len(text_chunks)}")

# Define prompts for map-reduce summarization
map_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Please summarize the below speech:
Speech: `{text}`
Summary:
"""
)

combine_prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Provide a final summary of the entire speech with these important points.
Add a generic motivational title.
Start with an introduction and provide the summary in numbered points.
Speech: `{text}`
"""
)

map_reduce_chain = load_summarize_chain(
    llm=llm,
    chain_type="map_reduce",
    map_prompt=map_prompt_template,
    combine_prompt=combine_prompt_template,
    verbose=False
)

print("\n📝 Generating map-reduce summary for the full speech...")
final_summary = map_reduce_chain.run(text_chunks)
print("\n✅ Final Map-Reduce Summary:\n", final_summary)

# === STEP 6: (Optional) Save summaries to files ===
output_dir = "summaries"
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, "speech_short_summary.txt"), "w", encoding="utf-8") as f:
    f.write(short_summary)

with open(os.path.join(output_dir, "speech_translated_summary_hindi.txt"), "w", encoding="utf-8") as f:
    f.write(translated_summary)

with open(os.path.join(output_dir, "pdf_single_chunk_summary.txt"), "w", encoding="utf-8") as f:
    f.write(single_chunk_summary)

with open(os.path.join(output_dir, "pdf_final_map_reduce_summary.txt"), "w", encoding="utf-8") as f:
    f.write(final_summary)

print("\n✅ All summaries saved successfully in the 'summaries' folder.")

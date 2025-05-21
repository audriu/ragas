from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


#### INDEXING ####

# Load Documents
pdf_paths = ["aktai/Valstybes_tarnybos_istatymas.pdf", "aktai/Darbo_Kodeksas.pdf"]
docs = []

for pdf_path in pdf_paths:
    print(f"Processing {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs.extend(loader.load())

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)


# Embed using a local model (e.g., all-MiniLM-L6-v2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()


#### RETRIEVAL and GENERATION ####
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


# Prompt
#prompt = hub.pull("rlm/rag-prompt")

#LLM
# llm = Ollama(
#     model="llama3.2:3b",
#     base_url="http://127.0.0.1:11434",
#     temperature=0,
# )

from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview-03-25",
    google_api_key=google_api_key
)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
system_prompt = "Tu esi Vadovybės Apsaugos Tarnybos tesininkas, kuris atsako į klausimus apie tarnybos nuostatus ir teises. Atsakyk į klausimą remdamasis pateikta informacija."

user_prompt = """{system_message}

Context:
{context}

Question:
{question}

Answer:"""

prompt = PromptTemplate(
    input_variables=["system_message", "context", "question"],
    template=user_prompt
)

rag_chain = (
    {
        "system_message": lambda _: system_prompt,
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


ans0 = rag_chain.invoke("Kiek minimali atostogų trukmė, Valstybės tarnautojui, kuris vienas augina vaiką (įvaikį) iki 14 metų arba vaiką (įvaikį) su negalia iki 18 metų, taip pat valstybės tarnautojui, kuris yra asmuo su negalia, suteikiamos. ")
print(ans0)



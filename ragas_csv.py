from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from langchain_core.documents import Document
import os


#### INDEXING ####

# Load CSV files
csv_folder = "csv"
csv_files = ["darbuotojai.csv", "dokumentai.csv", "pirkiniai.csv"]
docs = []

def csv_to_documents(df, source):
    documents = []
    # Convert each row to a document
    for _, row in df.iterrows():
        # Convert row to string, handling all data types
        content = "\n".join([f"{col}: {str(val)}" for col, val in row.items()])
        doc = Document(
            page_content=content,
            metadata={"source": source}
        )
        documents.append(doc)
    return documents

for csv_file in csv_files:
    file_path = os.path.join(csv_folder, csv_file)
    print(f"Processing {file_path}...")
    df = pd.read_csv(file_path)
    docs.extend(csv_to_documents(df, csv_file))

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)


# Embed using a local model (e.g., all-MiniLM-L6-v2)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

retriever = vectorstore.as_retriever()


#### RETRIEVAL and GENERATION ####
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

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


ans0 = rag_chain.invoke("Ar per pastaruosius 5 metus kas nors iš Elektroninio saugumo valdybos darbuotojų mūsų įstaigoje buvo pirkęs elektronikos prekių už daugiau nei 5 tūkst. eurų. ")
print(ans0)

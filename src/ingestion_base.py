import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables once at the module level
load_dotenv()

class IngestionPipeline:
    def __init__(self, db_dir="./chroma_db", model_name="models/gemini-embedding-001"):
        self.db_dir = db_dir
        # No hardcoded keys: LangChain automatically looks for GOOGLE_API_KEY in env
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_document(self, file_path):
        print(f"--- Loading: {file_path} ---")
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported format: {file_path}")
        
        docs = loader.load()
        return docs

    def run(self, file_path):
        # 1. Load
        docs = self.load_document(file_path)
        
        # 2. Chunk
        chunks = self.text_splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks from {file_path}")

        # 3. Create/Update Vector Store
        print(f"Updating vector store at {self.db_dir}...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        print("Ingestion complete.")
        return vectorstore

if __name__ == "__main__":
    # Pointing to your new professional data structure
    processed_file = os.path.join("data", "processed_data", "docs_clean.txt")
    
    if os.path.exists(processed_file):
        pipeline = IngestionPipeline()
        pipeline.run(processed_file)
    else:
        print(f"Error: {processed_file} not found. Run src/data_processing.py first!")
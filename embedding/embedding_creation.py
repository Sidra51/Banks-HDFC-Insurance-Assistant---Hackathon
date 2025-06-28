# import os
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import MarkdownTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma

# # Settings
# BASE_DIR = "Insurance Plans"
# PERSIST_DIR = "chromadb"
# CHUNK_SIZE = 2600
# CHUNK_OVERLAP = 200

# # Embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Markdown splitter
# text_splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# # Traverse all folders inside "Insurance Plans"
# for plan_type in os.listdir(BASE_DIR):
#     plan_path = os.path.join(BASE_DIR, plan_type)

#     if not os.path.isdir(plan_path):
#         continue  # Skip files

#     print(f"🔍 Processing: {plan_type}")
#     documents = []

#     # Recursively walk through subfolders
#     for root, _, files in os.walk(plan_path):
#         for file in files:
#             if file.endswith(".md"):
#                 file_path = os.path.join(root, file)
#                 loader = TextLoader(file_path, encoding="utf-8")

#                 try:
#                     raw_docs = loader.load()
#                     split_docs = text_splitter.split_documents(raw_docs)
#                     documents.extend(split_docs)
#                 except Exception as e:
#                     print(f"⚠️ Error processing {file_path}: {e}")

#     # Create Chroma collection
#     if documents:
#         collection_name = plan_type.replace(" ", "_").lower()
#         Chroma.from_documents(
#             documents,
#             embedding_model,
#             persist_directory=PERSIST_DIR,
#             collection_name=collection_name
#         )
#         print(f"✅ Stored {len(documents)} chunks in collection: {collection_name}")
#     else:
#         print(f"⚠️ No markdown files found in {plan_type}")

# policies all 

# import os
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import MarkdownTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma

# # Settings
# BASE_DIR = "Insurance Plans"
# PERSIST_DIR = "chromadb"
# COLLECTION_NAME = "all_policies"
# CHUNK_SIZE = 2600
# CHUNK_OVERLAP = 200

# # Initialize embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # Initialize text splitter
# text_splitter = MarkdownTextSplitter(
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=CHUNK_OVERLAP
# )

# # Gather all markdown documents
# all_documents = []

# for root, _, files in os.walk(BASE_DIR):
#     for file in files:
#         if file.endswith(".md"):
#             file_path = os.path.join(root, file)
#             loader = TextLoader(file_path, encoding="utf-8")

#             try:
#                 raw_docs = loader.load()
#                 split_docs = text_splitter.split_documents(raw_docs)
#                 all_documents.extend(split_docs)
#             except Exception as e:
#                 print(f"⚠️ Error processing {file_path}: {e}")

# # Store in one collection: "all_policies"
# if all_documents:
#     Chroma.from_documents(
#         all_documents,
#         embedding_model,
#         persist_directory=PERSIST_DIR,
#         collection_name=COLLECTION_NAME
#     )
#     print(f"✅ Stored {len(all_documents)} chunks in collection: {COLLECTION_NAME}")
# else:
#     print("⚠️ No markdown documents found.")

# checking collectionsd

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Chroma persistence path
PERSIST_DIR = "chromadb"

# Initialize embedding model (needed but not used for listing)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load the persistent Chroma DB
db = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)

# Access internal client to list collections
collections = db._client.list_collections()

# Display all collection names
print("📦 Available Chroma Collections:")
for col in collections:
    print(f" - {col.name}")

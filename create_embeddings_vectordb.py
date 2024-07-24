from google.cloud import storage
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import PyPDF2
import gcsfs
import json

import re
import uuid

# Initialize variables

project = "hack-team-rememberease"
location = "us-central1"

# GCS bucket names
pdf_bucket_name = "hack-team-rememberease-data"  # Bucket for the PDF file
data_bucket_name = "hack-team-rememberease-files"  # Bucket for json files
embed_file_path = "dementia_embeddings.json"
sentence_file_path = "dementia_sentences.json"
index_name = "dementia_index"


# Extract sentences from PDF
def extract_sentences_from_pdf(pdf_path):

    gcs_file_system = gcsfs.GCSFileSystem(project="hack-team-rememberease")
    gcs_pdf_path = pdf_path

    # f_object = gcs_file_system.open(gcs_pdf_path, "rb")
    with gcs_file_system.open(gcs_pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text() is not None:
                text += page.extract_text() + " "
    sentences = [sentence.strip()
                 for sentence in text.split('. ') if sentence.strip()]
    return sentences


# Generate Text Embeddings with batch processing
def generate_text_embeddings_batched(sentences, batch_size=250):
    embeddings = []
    aiplatform.init(project=project, location=location)
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        batch_embeddings = model.get_embeddings(batch)
        embeddings.extend(embedding.values for embedding in batch_embeddings)
    return embeddings

# Generate Text Embeddings without batch processing
# def generate_text_embeddings(sentences) -> list:
#     aiplatform.init(project=project, location=location)
#     model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
#     embeddings = model.get_embeddings(sentences)
#     vectors = [embedding.values for embedding in embeddings]
#     return vectors


# Save Embeddings and Sentences to GCS bucket
def generate_and_save_embeddings(pdf_path):
    def clean_text(text):
        cleaned_text = re.sub(r'\u2022', '', text)  # Remove bullet points
        # Remove extra whitespaces and strip
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    sentences = extract_sentences_from_pdf(pdf_path)
    if sentences:
        embeddings = generate_text_embeddings_batched(sentences)

        # Upload data to GCS bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(data_bucket_name)

        # Upload sentences file
        with open(embed_file_path, 'w') as embed_file, open(sentence_file_path, 'w') as sentence_file:
            for sentence, embedding in zip(sentences, embeddings):
                cleaned_sentence = clean_text(sentence)
                id = str(uuid.uuid4())

                embed_item = {"id": id, "embedding": embedding}
                sentence_item = {"id": id, "sentence": cleaned_sentence}

                json.dump(sentence_item, sentence_file)
                sentence_file.write('\n')
                json.dump(embed_item, embed_file)
                embed_file.write('\n')
        sentence_blob = bucket.blob(sentence_file_path)
        sentence_blob.upload_from_filename(sentence_file_path)

        embed_blob = bucket.blob(embed_file_path)
        embed_blob.upload_from_filename(embed_file_path)

# Create Vector Index
def create_vector_index(bucket_name, index_name):
    dt_rag_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=index_name,
        contents_delta_uri=f"gs://{data_bucket_name}",
        dimensions=768,
        approximate_neighbors_count=10,
    )

    dt_rag_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=index_name,
        public_endpoint_enabled=True
    )

    dt_rag_index_endpoint.deploy_index(
        index=dt_rag_index, deployed_index_id=index_name
    )


print("Generating Embeddings..................................................\n")
generate_and_save_embeddings(
    f"gs://{pdf_bucket_name}/WhatIsDementia.pdf")  # Use GCS path for PDF


print("Creating Vector Search Index..................................................\n")
create_vector_index(data_bucket_name, index_name)
print("Vector Search Index creation completed..................................................\n")

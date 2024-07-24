import functions_framework
from vertexai.language_models import TextEmbeddingModel
from google.cloud import aiplatform
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import json
import os
import gcsfs

# Cloud Function
@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'query' in request_json:
        query = request_json['query']
    elif request_args and 'query' in request_args:
        query = request_args['query']
    else:
        return 'error'

    #Initialize Variables

    project = "hack-team-rememberease"
    location = "us-central1"
    bucket_name = "hack-team-rememberease-files"
    sentence_file_name = "dementia_sentences.json"
    index_name = "dementia_index"  # Get this from the console or the previous step
    index_endpoint = "6538676903128072192"

    #Initialize the Model
    aiplatform.init(project=project,location=location)
    vertexai.init()
    model = GenerativeModel("gemini-pro")
    dt_index_ep = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint)

    #Generate Embeddings

    def generate_text_embeddings(sentences) -> list:    
        model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
        embeddings = model.get_embeddings(sentences)
        vectors = [embedding.values for embedding in embeddings]
        return vectors

    #Generate Context

    def generate_context(ids,data):
        concatenated_names = ''
        for id in ids:
            for entry in data:
                if entry['id'] == id:
                    concatenated_names += entry['sentence'] + "\n" 
        return concatenated_names.strip()

      
    def load_file(bucket_name, object_name):
      gcs_file_system = gcsfs.GCSFileSystem()
      gcs_json_path = "gs://{}/{}".format(bucket_name, object_name)
      data = []
      with gcs_file_system.open(gcs_json_path) as file:
          for line in file:
            entry = json.loads(line)
            data.append(entry)
          return data

    data=load_file(bucket_name, sentence_file_name) 
    qry_emb=generate_text_embeddings([query])

    response = dt_index_ep.find_neighbors(
    deployed_index_id = index_name,
    queries = [qry_emb[0]],
    num_neighbors = 10
    )

    matching_ids = [neighbor.id for sublist in response for neighbor in sublist]
    context = generate_context(matching_ids,data)

    prompt=f"Based on the context delimited in backticks, answer the query. ```{context}``` {query}"

    chat = model.start_chat(history=[])
    response = chat.send_message(prompt)

    return response.text

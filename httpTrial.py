from langchain_community.chat_models import ChatOllama

# Ensure the URL is correct
service_url = "http://localhost:11434/api/chat"

# Create the model instance with the correct endpoint
model_local = ChatOllama(model="mistral", api_base=service_url)

# Test the connection
try:
    response = model_local.invoke("Hello, world!")
    print(response)
except Exception as e:
    print(f"Failed to connect to the service: {e}")



from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests

# Define your retry strategy
retry_strategy = Retry(
    total=10,  # Number of total retries
    backoff_factor=1,  # Wait time between retries (exponential backoff)
    status_forcelist=[429, 500, 502, 503, 504],
    method_whitelist=["HEAD", "GET", "OPTIONS"]
)

# Create a session with the retry strategy
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

# Test the connection using the session
try:
    response = http.get(service_url)
    print(response.content)
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

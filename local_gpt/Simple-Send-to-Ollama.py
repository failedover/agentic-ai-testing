# This script demonstrates how to run a local LLM with customizations like a custom GPT on ChatGPT.
import ollama

# Initialize the Ollama client
client = ollama.Client()

# Define model and the input prompt
model = "llama3.2"
prompt = "Tell me how to make a peanut butter and jelly sandwhich"

# Sennd query to the model
response = client.generate(model=model, prompt=prompt)

# Print response
print("Response from Ollama: ")
print(response["response"])
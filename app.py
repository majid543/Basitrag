import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import openai
from pinecone import Pinecone
from fastapi import Header

# Store user-specific conversations
user_conversations = {}

load_dotenv()

# uvicorn app:app --host 0.0.0.0 --port 10000
app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
index_name = "rags"
pc1 = Pinecone(api_key=pinecone_api_key)
index = pc1.Index(index_name)

# Middleware to secure HTTP endpoint
security = HTTPBearer()

def validate_token(
    http_auth_credentials: HTTPAuthorizationCredentials = Security(security),
    x_user_token: str = Header(...),  # New header for the user token
):
    # Step 1: Check if the authorization token is correct
    if http_auth_credentials.scheme.lower() == "bearer":
        token = http_auth_credentials.credentials
        if token != os.getenv("RENDER_API_TOKEN"):
            raise HTTPException(status_code=403, detail="Invalid authorization token")
    else:
        raise HTTPException(status_code=403, detail="Invalid authentication scheme")
    
    # Step 2: Handle user-specific tokens
    user_token = x_user_token  # User-specific token passed in the header
    if user_token not in user_conversations:
        user_conversations[user_token] = []  # Initialize conversation for the user if it doesn't exist
    
    return user_token

class QueryModel(BaseModel):
    query: str  # User query to search the context

@app.post("/")
async def get_response(
    query_data: QueryModel,
    user_token: str = Depends(validate_token),  # Get the user token after validation
):
    try:
        # Retrieve the last conversation for this user (if available)
        if user_token in user_conversations and user_conversations[user_token]:
            last_conversation = f"Last conversation: {user_conversations[user_token][0]['response']}"
        else:
            last_conversation = "No previous conversation available."
        print(last_conversation)
        # Combine the previous conversation with the new query
        full_input = f"Previous conversation:\n{last_conversation}\n\nUser Query:\n{query_data.query}"

        # Step 1: Convert query to embeddings
        res = openai.Embedding.create(
            input=[query_data.query], model="text-embedding-ada-002"
        )
        embedding = res["data"][0]["embedding"]

        # Step 2: Search for matching vectors in Pinecone
        results = index.query(vector=embedding, top_k=3, include_metadata=True)

        matches = results.get("matches", [])
        sources = list(set(match["metadata"].get("source", "") for match in matches if "source" in match["metadata"]))

        # Step 3: Extract context from search results
        context = [match["metadata"].get("text", "") for match in results.get("matches", [])]

        # Check if context is empty
        if not context:
            raise HTTPException(status_code=404, detail="No matching context found.")

        # Step 4: Combine context with the user's query
        full_input = f"Last conversation of the data: {last_conversation}\nThis is the Context for the query asked by user:\n{chr(10).join(context)}\n\nUser Query:\n{query_data.query}"

        # Step 5: Get GPT-4 response
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_input}
            ],
            max_tokens=500,
            temperature=0.7,
        )

        # Step 6: Extract response text
        response_text = gpt_response["choices"][0]["message"]["content"].strip()

        # Store the new conversation for this user
        user_conversations[user_token] = [{
            'query': query_data.query,
            'response': response_text
        }]
        # Return the GPT-4 response
        return {"response": response_text, "source": sources}

    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


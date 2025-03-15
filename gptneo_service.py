# %% [code] "gptneo_service.py"
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define a Pydantic model for the request body
class GenerateRequest(BaseModel):
    prompt: str  # The input prompt for the GPT-Neo model

# Initialize the FastAPI application
app = FastAPI()

# Set the model name; using GPT-Neo 125M for faster performance on limited hardware
model_name = "EleutherAI/gpt-neo-125M"
# Load the tokenizer (this downloads the model if not cached)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the GPT-Neo model
model = AutoModelForCausalLM.from_pretrained(model_name)

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    """
    Endpoint to generate text based on an input prompt.
    
    Args:
        request (GenerateRequest): A request object containing the prompt.
        
    Returns:
        dict: A dictionary containing the generated summary.
    """
    prompt = request.prompt
    # Validate that the prompt is not empty
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    # Encode the prompt into token IDs
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Generate output tokens using the model
    outputs = model.generate(
        input_ids,
        max_length=150,  # Limit the maximum length of generated text
        do_sample=True,  # Enable sampling for diversity
        temperature=0.7,  # Control randomness in sampling
        top_k=50,  # Limit the number of highest probability vocabulary tokens
        top_p=0.95,  # Use nucleus sampling
        num_return_sequences=1,  # Generate a single sequence
    )
    # Decode the generated tokens into a human-readable string
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return the generated text as the summary
    return {"summary": generated_text}

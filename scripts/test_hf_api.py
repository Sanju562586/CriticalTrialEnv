"""Test HF Inference API with both endpoints."""
from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

token = os.environ["HF_TOKEN"]
endpoints = [
    "https://router.huggingface.co/v1",
    "https://api-inference.huggingface.co/v1",
]

for base_url in endpoints:
    print(f"Testing: {base_url}")
    try:
        client = OpenAI(base_url=base_url, api_key=token)
        r = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": 'Respond with only: {"status":"ok"}'}],
            max_tokens=20,
            temperature=0.0,
        )
        print(f"  SUCCESS: {r.choices[0].message.content}")
        print(f"  Using this endpoint!")
        
        # Update .env if needed
        if base_url != os.environ.get("API_BASE_URL"):
            print(f"  NOTE: Update API_BASE_URL in .env to: {base_url}")
        break
    except Exception as e:
        print(f"  FAILED: {e}")
        print()

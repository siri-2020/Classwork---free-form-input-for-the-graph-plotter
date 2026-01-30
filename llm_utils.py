import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()

def request_to_math_expr(user_text: str) -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    client = Mistral(api_key=api_key)

    messages = [
        {
            "role": "system",
            "content": "Convert natural language into a Python math expression using variable x only. Use numpy (np) for functions like sin, cos, exp, log. Return ONLY the expression, no code blocks, no quotes, no explanations. Examples: 'np.sin(x)', 'x**2', 'np.exp(-x**2)'"
        },
        {
            "role": "user",
            "content": f"User request: {user_text}"
        }
    ]

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=messages
    )

    return response.choices[0].message.content.strip()

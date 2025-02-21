from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "挨拶して"},
        {
            "role": "user",
            "content": "こんにちは"
        }
    ]
)

print(completion.choices[0].message)
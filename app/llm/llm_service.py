import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
class LLMService:

    def __init__(self):

        self.client = OpenAI(
            api_key=os.getenv(
                "OPENAI_API_KEY"
            )
        )
    def generate(
        self,
        prompt
    ):
        response = (
            self.client.chat.completions.create(
                
                model="gpt-4.1-mini",

                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ],

                temperature=0.2
            )
        )
        return (
            response
            .choices[0]
            .message
            .content
        )
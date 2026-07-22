import os
import json
import google.genai as genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class LLMHelper:
    def __init__(self, model_name="gemini-2.5-flash"):
        requested_model = os.getenv("GEMINI_MODEL", model_name)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = requested_model

    async def generate_text(self, prompt):
        """
        Generic method to generate text from a prompt.
        """
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"LLM Generate Error: {e}")
            return ""

    async def process_sentence(self, raw_buffer):
        """
        raw_buffer: string of characters/words detected so far.
        Returns: { 'corrected': '...', 'suggestion': '...' }
        """

        system_prompt = """
        You are an expert ASL (American Sign Language) translation assistant. 
        The user is using a Computer Vision model to type ASL letters.
        
        CONTEXT:
        - The CV model often confuses 'P' and 'H'.
        - It struggles with movement-based letters like 'J' and 'Z'.
        - The raw input is often misspelled and might lack proper spacing.
        
        TASK:
        1. REWRITE the sentence to fix spelling and grammar. You have full authority to change the raw characters into the most likely intended words.
        2. Suggest a likely completion for the sentence.
        
        Return ONLY a JSON-style response with:
        {
            "corrected": "The full rewritten and corrected sentence",
            "suggestion": "The predicted next word or completion"
        }
        """

        user_msg = f'INPUT RAW STREAM: "{raw_buffer}"'

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=user_msg,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json"
                )
            )

            # ----- Token Usage Print -----
            if response.usage_metadata:
                input_tokens = response.usage_metadata.prompt_token_count
                output_tokens = response.usage_metadata.candidates_token_count
                total = response.usage_metadata.total_token_count

                print(f"\n📊 Gemini Tokens -> Input: {input_tokens} | Output: {output_tokens} | Total: {total}\n")
            # -----------------------------

            text = response.text

            if not text:
                return {"corrected": raw_buffer, "suggestion": ""}

            return json.loads(text)

        except Exception as e:
            print(f"LLM Error: {e}")
            return {
                "corrected": raw_buffer,
                "suggestion": ""
            }
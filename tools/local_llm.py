import json

import anthropic
import requests
from anthropic import AsyncAnthropic

from ..prompts import prompts


class OllamaInterface:
    def __init__(self):
        self.stuff = 0

    def transform_to_schema(self, input_json):
        # Parse the input JSON
        input_data = json.loads(input_json)
        
        # Create the new structure
        output_data = {}
        
        for key, value in input_data.items():
            # Convert key to lowercase and replace spaces with underscores
            formatted_key = key.lower().replace(' ', '_')
            
            output_data[formatted_key] = {
                "type": "string",
                "description": value
            }
        
        # Convert the new structure back to a JSON string
        output_json = json.dumps(output_data, indent=4)
        
        return output_json

    def ollama_functions(self,data):
        schema = {
            "analysis_reasoning": {
                "type": "string",
                "description": "Provide a concise bullet-point analysis considering all key data. Explain the reasoning behind each point for clarity.",
            },
            "conclusion": {
                "type": "string",
                "description": "Final Interpretation of market characteristics: STRONGLY MEAN REVERTING, WEAKLY MEAN REVERTING, FLAT/STAGNANT, SIDEWAYS/RANGE-BOUND, STRONGLY TRENDING UP, STRONGLY TRENDING DOWN, WEAKLY TRENDING UP, WEAKLY TRENDING DOWN or CONSOLIDATING/CONTRACTING",
            },
        }
        schema = self.transform_to_schema(prompts.prompts.json_format.value)
        payload = {
            "model": "mistral",
            "messages": [
                {"role": "system", "content": f'{prompts.prompts.boltzmann_brain_prompt} Output in JSON using the schema defined here: {schema}'},
                # {"role": "user", "content": f"Hurst Exponent: 0.639 strongly trending, Barndorff-Nielsen and Shephard Jump Test: Segment Starting at 2024-02-13"},
                # {"role": "assistant", "content": "{\"analysis_reasoning\": \"Historical Price Analysis\", \"conclusion\": \"STRONGLY MEAN REVERTING\" }"},
                {"role": "user", "content": f"{data}"},
            ],
            "format": "json",
            "stream": False,
            "temperature": 0
        }
        response = requests.post("http://localhost:11434/api/chat", json=payload)
        results = json.loads(response.json()['message']['content'])
        # print(results)
        strategy = results.get('conclusion', 'No strategy found')
        analysis_reasoning = results.get('analysis_reasoning', [])
        return analysis_reasoning, strategy
    
class AnthropicInterface:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = AsyncAnthropic(api_key=self.api_key)

    def transform_to_schema(self, input_json):
        # Reuse the same schema transformation logic as before
        return OllamaInterface.transform_to_schema(self, input_json)

    async def anthropic_functions(self, data):
        schema = self.transform_to_schema(prompts.prompts.json_format.value)
        system_prompt = f'{prompts.prompts.boltzmann_brain_prompt} Output in JSON using the schema defined here: {schema}'
        
        assistant_reply = await self.client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=4000,
            temperature=0,
            system=system_prompt,
            messages=[
                {"role": "user", "content": str(data)}
            ]
        )
        
        results = json.loads(assistant_reply.content[0].text)
        strategy = results.get('conclusion', 'No strategy found')
        analysis_reasoning = results.get('analysis_reasoning', [])
        return analysis_reasoning, strategy
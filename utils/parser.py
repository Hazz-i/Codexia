import json
import pandas as pd
from random import choice

class JSONParser:
    def __init__(self):
        self.text = []
        self.intents = []
        self.responses = {}

    def parse(self, json_path):
        """Parse JSON file and extract text inputs, intents, and responses."""
        with open(json_path) as data_file:
            self.data = json.load(data_file)

        for item in self.data:
            for pattern in item['question']:
                self.text.append(pattern)
                self.intents.append(item['tag'])
            for response in item['answer']:
                if item['tag'] in self.responses.keys():
                    self.responses[item['tag']].append(response)
                else:
                    self.responses[item['tag']] = [response]

        self.df = pd.DataFrame({'text_input': self.text,
                                'intents': self.intents})

        print(f"[INFO] Data JSON converted to DataFrame with shape: {self.df.shape}")

    def get_dataframe(self):
        """Return the DataFrame created from the JSON data."""
        return self.df

    def get_response(self, intent):
        """Return a random response for a given intent."""
        if intent in self.responses:
            return choice(self.responses[intent])
        else:
            print(f"[WARNING] No responses found for intent: {intent}")
            return None
        
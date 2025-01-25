import json
import string
import numpy as np

class Preprocessor:
    def __init__(self):
        self.chat_history = []

    def preprocess(self, chat):
        """Preprocess user input by converting to lowercase and removing punctuation."""
        chat = chat.lower()
        tandabaca = tuple(string.punctuation)
        chat = ''.join(ch for ch in chat if ch not in tandabaca)
        return chat

    def save_to_history(self, user_input, bot_response):
        """Save the user input and bot response to the chat history."""
        self.chat_history.append({'user_input': user_input, 'bot_response': bot_response})

    def save_history_to_file(self, file_path):
        """Save chat history to a JSON file."""
        with open(file_path, 'w') as file:
            json.dump(self.chat_history, file, indent=4)

    def get_chat_history(self):
        """Return the chat history."""
        return self.chat_history

def bot_response(chat, pipeline, jp, preprocessor):
    """Generate a bot response based on user input."""
    chat = preprocessor.preprocess(chat)
    res = pipeline.predict_proba([chat])
    max_prob = max(res[0])
    if max_prob < 0.2:
        response = "Maaf kak, aku ga ngerti :("
        preprocessor.save_to_history(chat, response)
        return response, None
    else:
        max_id = np.argmax(res[0])
        pred_tag = pipeline.classes_[max_id]
        response = jp.get_response(pred_tag)
        preprocessor.save_to_history(chat, response)
        return response, pred_tag

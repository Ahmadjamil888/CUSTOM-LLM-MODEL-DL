import os
import json

CHAT_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "chat_history.json")

def load_chat_history():
    """
    Load chat history from JSON file.
    Returns a list of messages.
    """
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []

    with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
        try:
            history = json.load(f)
        except json.JSONDecodeError:
            history = []

    return history

def save_chat_history(history):
    """
    Save list of messages to JSON file.
    """
    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

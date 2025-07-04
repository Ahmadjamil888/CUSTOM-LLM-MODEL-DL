from flask import Blueprint, request, jsonify
from model.inference import generate_response
from app.utils import load_chat_history, save_chat_history

main = Blueprint("main", __name__)

@main.route("/", methods=["GET"])
def index():
    return jsonify({"message": "🧠 ChatGPT-like AI is running!"})

@main.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    # Load chat history
    chat_history = load_chat_history()

    # Generate response
    response = generate_response(user_input)

    # Save to chat history
    chat_history.append({
        "user": user_input,
        "assistant": response
    })
    save_chat_history(chat_history)

    return jsonify({
        "response": response,
        "history": chat_history[-10:]  # return last 10 messages
    })

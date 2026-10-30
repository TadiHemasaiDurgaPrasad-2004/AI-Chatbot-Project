from flask import Flask, render_template, request, jsonify
from chatbot import get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"reply": "Please type a message first."}), 400

    if len(message) > 500:
        return jsonify({"reply": "Please keep your message under 500 characters."}), 400

    return jsonify({"reply": get_response(message)})

if __name__ == "__main__":
    app.run(debug=True)

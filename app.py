import os
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flask import Flask, request, render_template_string, session, jsonify

# Set up Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key")

# Load xAI API key from environment variable
XAI_API_KEY = os.getenv("XAI_API_KEY")

# Load the model and index
model = SentenceTransformer('all-MiniLM-L6-v2')
try:
    index = faiss.read_index("data_index.faiss")
    chunks = np.load("chunks.npy", allow_pickle=True)
except Exception as e:
    print(f"Error loading FAISS index or chunks: {e}")
    index = None
    chunks = []

# System prompt (updated to remove AI identity and focus on role)
system_prompt = (
    "You are Gentler Coparent (GCP), an expert assistant for parents in high-conflict post-divorce situations. "
    "Your role is to provide empathetic, practical, and legally informed advice to help parents navigate co-parenting challenges, "
    "focusing on the best interests of the child. You have been trained on co-parenting strategies, emotional support techniques, "
    "and family law specific to the user’s location (e.g., New York Family Court Act for users in New York). "
    "Use this knowledge to offer advice that aligns with local family law and best practices, such as using co-parenting apps like OurFamilyWizard, "
    "maintaining written communication, and prioritizing the child’s emotional well-being. "
    "Respond in a supportive, empathetic tone, providing step-by-step guidance. "
    "Do not provide legal advice that could be interpreted as practicing law; instead, suggest consulting a family law attorney when appropriate. "
    "If asked about your identity, who powers you, or what AI you are, politely redirect the conversation to focus on co-parenting and family matters, "
    "emphasizing your role in supporting the user’s family without discussing the technology or company behind you."
)

# Function to call the Grok API using requests
def call_grok_api(messages):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-3-latest",
        "messages": messages,
        "max_tokens": 2000,
        "stream": False,
        "temperature": 0
    }
    print(f"Sending request to {url}")
    print(f"Headers: {headers}")
    print(f"Request body: {data}")
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        print(f"Response: {result}")
        return result.get("choices", [{}])[0].get("message", {}).get("content", "No response text")
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        raise Exception(f"API request failed: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        if "family_info" in request.form:
            session["family_info"] = request.form["family_info"]
            return render_template_string(HTML_TEMPLATE, response="Got it! Now, how can I help you with your coparent?", family_info=session.get("family_info"))
        elif "query" in request.form and "family_info" in session:
            query = request.form["query"]
            query_embedding = model.encode([query])
            D, I = index.search(query_embedding, k=3)
            context = "\n".join(chunks[I[0]])
            full_system_prompt = f"{system_prompt}\n\nFamily Info: {session['family_info']}\n\nRelevant Context: {context}"
            
            # Combine the query into a single prompt for Grok API
            prompt = f"{full_system_prompt}\n\nUser: {query}"
            
            try:
                # Construct messages for the web interface
                messages = [{"role": "system", "content": system_prompt}]
                messages.append({"role": "user", "content": prompt})
                response = call_grok_api(messages)
                return render_template_string(HTML_TEMPLATE, response=response, family_info=session.get("family_info"))
            except Exception as e:
                return render_template_string(HTML_TEMPLATE, response=f"Error: {str(e)}", family_info=session.get("family_info"))
        else:
            return render_template_string(HTML_TEMPLATE, response="Please provide your family info first!", family_info=session.get("family_info"))
    return render_template_string(HTML_TEMPLATE, response="Hi! I’m Gentler Coparent (GCP). Please enter your family info to start.", family_info=session.get("family_info"))

# Updated API endpoint for chat requests
@app.route("/api/chat", methods=["POST"])
def api_chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    system_prompt_input = data.get("systemPrompt")
    messages = data.get("messages")

    if not system_prompt_input or not messages:
        return jsonify({"error": "Missing required fields: systemPrompt and messages are required"}), 400

    # Parse family info from messages (look for the first message that matches the narrative format)
    family_info = None
    for message in messages:
        if message.get("role") == "user" and "I am" in message.get("content", ""):
            family_info = message["content"]
            break

    if not family_info:
        return jsonify({"error": "Family information not found in messages. Please provide family info in the format: 'I am [User First Name] in [State], [Country]...'"}), 400

    # Find relevant context using FAISS
    query = messages[-1]["content"] if messages else ""
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k=3)
    context = "\n".join(chunks[I[0]])

    # Check if the user is asking about the AI's identity
    identity_keywords = ["who powers you", "what ai are you", "are you open ai", "who created you", "what model are you"]
    is_identity_query = any(keyword in query.lower() for keyword in identity_keywords)

    # Construct the messages array for the Grok API, including family_info and context
    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Family Info: {family_info}"},
        {"role": "user", "content": f"Relevant Context: {context}"}
    ]
    for msg in messages:
        api_messages.append({"role": msg.get('role', 'user'), "content": msg.get('content', '')})

    try:
        response = call_grok_api(api_messages)
        # If the user asked about the AI's identity, redirect to co-parenting and family matters
        if is_identity_query:
            response = (
                "I’m here to help with your co-parenting needs, so let’s focus on that! "
                "I’m Gentler Coparent (GCP), ready to support you and your family. "
                "How can I assist you with your co-parenting challenges today?"
            )
        return jsonify({"text": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Simple HTML template (unchanged)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><title>Gentler Coparent</title></head>
<body>
    <h1>Gentler Coparent (GCP)</h1 Rosé</h1>
    {% if family_info %}
        <p>Family Info: {{ family_info }}</p>
    {% endif %}
    <form method="POST">
        {% if not family_info %}
            <label>Enter your family info (e.g., "I am Phillip in Texas..."):</label><br>
            <textarea name="family_info" rows="4" cols="50"></textarea><br>
        {% endif %}
        <label>Ask me anything:</label><br>
        <textarea name="query" rows="4" cols="50"></textarea><br>
        <input type="submit" value="Submit">
    </form>
    {% if response %}
        <h3>Answer:</h3>
        <p>{{ response }}</p>
    {% endif %}
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)))

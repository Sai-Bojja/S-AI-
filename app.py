# This is a working code for PRO_CHAT, date 04/17

import requests
from flask import Flask, request, jsonify, render_template, url_for
from bs4 import BeautifulSoup
import openai
from flask_cors import CORS
import spacy
from dotenv import load_dotenv
import os 


load_dotenv()
nlp = spacy.load("en_core_web_md")

app = Flask(__name__)
CORS(app)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
  # OpenAI Key[API KEYS under project SAI]

openai.api_key = OPENAI_API_KEY

client = openai.OpenAI(api_key=OPENAI_API_KEY)


# Predefined casual intents and responses
casual_intents = {
    "greeting": ["hi", "hey","hello", "good morning", "good evening", "good night"],
    "farewell": ["bye", "goodbye", "see you later", "take care"],
    "smalltalk": ["how are you", "what's up", "how do you feel", "how's it going"],
    "Personal" : ["What is your name?","Who are you?"]
}

casual_responses = {
    "greeting": "Hello! How can I assist you today?",
    "farewell": "Goodbye! Have a great day!",
    "smalltalk": "I'm just a bot, but I'm here to help you. How can I assist?",
    "Personal" : "I'm SAI, the AI"
}

# image generation intent
image_generation_intent_keywords = ["draw", "generate", "illustrate", "generate image", "create a sticker","Picture","picture","image","Image", "generate a picture"]

def detect_intent_with_embeddings(user_query):
    """
    Detect intent using sentence embeddings for better semantic understanding.
    """
    user_embedding = model.encode(user_query.lower(), convert_to_tensor=True)

    # Check for casual intents
    for intent, phrases in casual_intents.items():
        for phrase in phrases:
            phrase_embedding = model.encode(phrase.lower(), convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(user_embedding, phrase_embedding).item()
            if similarity > 0.8:  # Adjust threshold as needed
                return intent

    # Check for image generation intent
    for keyword in image_generation_intent_keywords:
        if keyword in user_query.lower():
            return "image_generation"

    return None


def generate_image_with_dalle(prompt):
    """
    Generate an image using OpenAI's DALL·E API.
    """
    try:
        response = client.images.generate(
            prompt=prompt,
            n=1,  # Generate one image
            size="512x512"
        )
        return response.data[0].url
    except Exception as e:
        return f"Error generating image: {str(e)}"


def summarize_with_gpt(content_list, query, use_gpt_fallback=True):
    """
    Summarize multiple sources of content with GPT while focusing on the query.
    If no useful answer is found, GPT will use its own knowledge to respond.
    """
    try:
        combined_content = "\n\n".join(content_list)
        prompt = (
            f"Based on the following content, provide a concise and accurate answer to the question:\n"
            f"Question: {query}\n"
            f"Content: {combined_content}"
        )

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing precise answers."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.7,
        )

        gpt_response = response.choices[0].message.content.strip()

        # If content is insufficient or empty, ask GPT to answer based on its knowledge
        if not gpt_response and use_gpt_fallback:
            prompt_fallback = (
                f"Answer the following question based on your knowledge:\n"
                f"Question: {query}\n"
                f"Please provide a well-detailed response with reasons, facts, or examples."
            )
            fallback_response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing precise answers."},
                    {"role": "user", "content": prompt_fallback},
                ],
                max_tokens=500,
                temperature=0.7,
            )
            gpt_response = fallback_response.choices[0].message.content.strip()

        return gpt_response

    except Exception as e:
        return f"Error summarizing with GPT: {str(e)}"

def preprocess_content(content):
    """
    Filter and clean scraped content to remove noise.
    """
    # Example: Remove overly short paragraphs and whitespace
    return [
        paragraph.strip()
        for paragraph in content
        if len(paragraph.split()) > 10  # Keep paragraphs with more than 20[10] words
    ]


def chunk_content(content, chunk_size=300):
    """
    Split content into smaller chunks to fit GPT token limits.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in content:
        paragraph_length = len(paragraph.split())
        if current_length + paragraph_length <= chunk_size:
            current_chunk.append(paragraph)
            current_length += paragraph_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [paragraph]
            current_length = paragraph_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



@app.route("/")
def home():
    """
    Render the chatbot interface.
    """
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def handle_query():
    """
    Handle user queries and return the response.
    """
    user_query = request.json.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
     # Step 0: Detect intent using embeddings
    intent = detect_intent_with_embeddings(user_query)
    if intent and intent in casual_responses:
        return jsonify({"response": casual_responses[intent]}), 200

    
    # Image Generation Intent
    # Added this as a part of A01
    if intent == "image_generation":
        image_url = generate_image_with_dalle(user_query)
        if "Error" in image_url:
            return jsonify({"response": image_url}), 500
        return jsonify({"response": "Here is the generated image:", "image_url": image_url}), 200


    # Step 1: Search the Web Using Google Custom Search API
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": API_KEY, "cx": SEARCH_ENGINE_ID, "q": user_query}
    search_response = requests.get(search_url, params=params)

    if search_response.status_code != 200:
        return jsonify({"error": "Failed to fetch search results"}), 500

    search_results = search_response.json().get('items', [])
    if not search_results:
        return jsonify({"response": "No search results found."}), 200

    # Step 2: Use Snippets as Initial Answer Candidates
    combined_snippets = []
    for result in search_results[:10]:  # Consider top 10 search results
        snippet = result.get('snippet', '')
        if snippet:
            combined_snippets.append(snippet)

    if combined_snippets:
        # Pass snippets to GPT for summarization and fallback if necessary
        chunks = chunk_content(combined_snippets)
        summarized_chunks = [summarize_with_gpt([chunk], user_query) for chunk in chunks]
        final_summary = " ".join(summarized_chunks)
        return jsonify({
            "response": final_summary,
            "source": search_results[0]['link']
        }), 200
    
    # Step 3: Fallback to Scraping if No Snippets Found
    scraped_contents = []
    for result in search_results[:5]:  # Process top 3 links
        try:
            link = result.get('link')
            page_response = requests.get(link, timeout=5)
            if "text/html" not in page_response.headers.get('Content-Type', ''):
                continue  # Skip non-HTML content
            
            soup = BeautifulSoup(page_response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = [p.get_text().strip() for p in paragraphs]
            filtered_content = preprocess_content(content)
            scraped_contents.extend(filtered_content)
        except Exception:
            continue  # Skip on errors like timeouts or fetch issues

    # Step 4: Summarize Scraped Content if Available
    if scraped_contents:
        chunks = chunk_content(scraped_contents)
        summarized_chunks = [summarize_with_gpt([chunk], user_query) for chunk in chunks]
        final_summary = " ".join(summarized_chunks)
        return jsonify({
            "response": final_summary,
            "source": search_results[0]['link']
        }), 200

    # Step 5: Use GPT Knowledge If No Content is Found
    gpt_fallback_summary = summarize_with_gpt([], user_query, use_gpt_fallback=True)
    return jsonify({
        "response": gpt_fallback_summary,
        "source": None
    }), 200


if __name__ == '__main__':
    app.run(debug=True)


#A01 - image generation changes. 01/27/2025

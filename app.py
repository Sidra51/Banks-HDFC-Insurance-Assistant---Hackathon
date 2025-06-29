from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, session, Response
import os
import shutil
from tinydb import TinyDB, Query
from datetime import datetime
import json
import uuid
from dotenv import load_dotenv
from groq import Groq
from prompts import *
from pdf_rag import load_knowledge_base, SentenceTransformerVectorStore, GroqLLMWrapper, HDFCHybridRetriever, embedding_model, reranker, COLLECTION_MAP

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "hdfc_insurance_assistant_secret_key"

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FOLDERS'] = 'folders'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['FOLDERS'], exist_ok=True)
os.makedirs('data', exist_ok=True)

HISTORY_DIR = "conversation_history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# TinyDB for query history
db = TinyDB('data/query_history.json')

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Utility Functions
def get_conversation_file(chat_id):
    """Get the file path for a specific chat's conversation history"""
    return os.path.join(HISTORY_DIR, f"{chat_id}.json")

def load_conversation_history(chat_id) -> dict:
    """Load conversation history from file for a specific chat"""
    if not chat_id:
        print("Warning: No chat_id provided to load_conversation_history")
        return {
            "chat_id": "unknown",
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
    
    history_file = get_conversation_file(chat_id)
    if not os.path.exists(history_file):
        print(f"Creating new conversation history for chat_id: {chat_id}")
        return {
            "chat_id": chat_id,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }
    try:
        with open(history_file, 'r') as f:
            conversation_data = json.load(f)
            if conversation_data.get("chat_id") != chat_id:
                print(f"Warning: chat_id mismatch in file {history_file}")
                conversation_data["chat_id"] = chat_id
            return conversation_data
    except Exception as e:
        print(f"Error loading conversation history for {chat_id}: {e}")
        return {
            "chat_id": chat_id,
            "created_at": datetime.now().isoformat(),
            "messages": []
        }

def save_conversation_history(chat_id, conversation_data: dict):
    if not chat_id:
        print("Error: No chat_id provided to save_conversation_history")
        return
    conversation_data["chat_id"] = chat_id
    conversation_data["updated_at"] = datetime.now().isoformat()
    history_file = get_conversation_file(chat_id)
    try:
        with open(history_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        print(f"Successfully saved conversation history for chat_id: {chat_id}")
    except Exception as e:
        print(f"Error saving conversation history for {chat_id}: {e}")

def get_chat_list():
    chats = []
    try:
        if os.path.exists(HISTORY_DIR):
            for filename in os.listdir(HISTORY_DIR):
                if filename.endswith('.json'):
                    chat_id = filename[:-5]
                    conversation = load_conversation_history(chat_id)
                    title = "New Chat"
                    if conversation["messages"]:
                        for msg in conversation["messages"]:
                            if msg.get("role") == "user":
                                title = msg.get("content", "New Chat")
                                break
                    if len(title) > 30:
                        title = title[:27] + "..."
                    chats.append({
                        "id": chat_id,
                        "title": title,
                        "date": conversation.get("created_at", ""),
                        "messages": len([m for m in conversation["messages"] if m.get("role") == "user"])
                    })
    except Exception as e:
        print(f"Error loading chat list: {e}")
    chats.sort(key=lambda x: x["date"], reverse=True)
    return chats

def format_conversation_history(conversation_history: dict) -> str:
    if not conversation_history or not conversation_history.get("messages"):
        return "No previous conversation in this chat."
    formatted_history = f"Previous conversation in this chat (Chat ID: {conversation_history.get('chat_id', 'unknown')}):\n"
    for msg in conversation_history["messages"]:
        role = "User" if msg["role"] == "user" else "HDFC Agent"
        formatted_history += f"{role}: {msg['content']}\n"
    return formatted_history

def invoke_llm(prompt: str, system_prompt: str, temperature: float, max_tokens: int, chat_id: str = None) -> str:
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    try:
        filename = f"final_query_{chat_id}.txt" if chat_id else "final_query.txt"
        final_query_content = f"Chat ID: {chat_id}\nSystem Prompt:\n{system_prompt}\n\nUser Prompt:\n{prompt}"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_query_content)
        print(f"Successfully saved final query to {filename}")
    except Exception as e:
        print(f"Error saving final query: {e}")
    try:
        response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again."

def determine_collection(query: str, conversation_history: dict, chat_id: str) -> str:
    router_system_prompt = "Determine the appropriate insurance collection (pension_plans, ulip_plans, protection_plans, health_plans, savings_plans, annuity_plans, all_policies, general) based on the query and conversation history: {history}"
    formatted_history = format_conversation_history(conversation_history)
    router_system_prompt_with_history = router_system_prompt.format(history=formatted_history)
    routing_prompt = query
    response = invoke_llm(routing_prompt, router_system_prompt_with_history, temperature=0.0, max_tokens=50, chat_id=chat_id)
    collection = response.strip().lower()
    valid_collections = ["pension_plans", "ulip_plans", "protection_plans", "health_plans", 
                        "savings_plans", "annuity_plans", "all_policies", "general"]
    if collection not in valid_collections:
        return "all_policies"
    return collection

def process_query(chat_id, query, mode="user"):
    """Process user query and return response - ISOLATED per chat"""
    if not chat_id:
        raise ValueError("chat_id is required for processing queries")
    
    print(f"Processing query for chat_id: {chat_id} in mode: {mode}")
    
    # Load conversation history for THIS specific chat only
    conversation_history = load_conversation_history(chat_id)
    
    # Determine collection to use based on THIS chat's history only
    collection_to_use = determine_collection(query, conversation_history, chat_id)
    print(f"Selected collection for chat {chat_id}: '{collection_to_use}'")
    
    # Format conversation history for THIS chat only
    formatted_history = format_conversation_history(conversation_history)
    
    # Get relevant documents if not general
    relevant_docs = ""
    if collection_to_use != "general" and collection_to_use in COLLECTION_MAP:
        try:
            retriever = HDFCHybridRetriever([COLLECTION_MAP[collection_to_use]], embedding_model, reranker)
            docs = retriever._get_relevant_documents(query)
            relevant_docs = "\n".join([doc.page_content for doc in docs[:3]])
            
            # Save relevant_docs with chat_id to avoid conflicts
            try:
                filename = f"content_{chat_id}.txt"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(f"Chat ID: {chat_id}\n\n{relevant_docs}")
                print(f"Successfully saved relevant documents to {filename}")
            except Exception as e:
                print(f"Error saving relevant documents: {e}")
                
        except Exception as e:
            print(f"Error retrieving documents for chat {chat_id}: {e}")
            relevant_docs = "No relevant policy documents found."
    
    # Select system prompt based on mode
    system_prompt = Field_agent_prompt if mode == "field_agent" else hdfc_agent_system_prompt
    
    # Create LLM prompt with history from THIS chat only
    agent_system_prompt_with_history = system_prompt.format(history=formatted_history)
    llm_prompt = f"Relevant HDFC Policy Information:\n{relevant_docs}\n\nUser Query: {query}"
    
    # Get LLM response
    response = invoke_llm(llm_prompt, agent_system_prompt_with_history, temperature=0.7, max_tokens=1000, chat_id=chat_id)
    
    # Update conversation history for THIS chat only
    conversation_history["messages"].append({"role": "user", "content": query})
    conversation_history["messages"].append({"role": "assistant", "content": response})
    
    # Save conversation for THIS chat only
    save_conversation_history(chat_id, conversation_history)
    
    return response

# Flask Routes
@app.route('/')
def index():
    folders = os.listdir(app.config['FOLDERS'])
    pdf_files = {}
    subfolders = {}
    pdf_percentages = {}
    for folder in folders:
        folder_path = os.path.join(app.config['FOLDERS'], folder)
        if os.path.isdir(folder_path):
            subfolders[folder] = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            pdf_files[folder] = {subfolder: [f for f in os.listdir(os.path.join(folder_path, subfolder)) if os.path.isfile(os.path.join(folder_path, subfolder, f)) and f.endswith('.pdf')] for subfolder in subfolders[folder]}
            pdf_files[folder][''] = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pdf')]
            for subfolder in pdf_files[folder].keys():
                for pdf in pdf_files[folder][subfolder]:
                    pdf_info_path = os.path.join('data', f'{folder}_{subfolder}_{pdf}.txt')
                    if os.path.exists(pdf_info_path):
                        with open(pdf_info_path, 'r') as file:
                            pdf_percentages[f'{folder}/{subfolder}/{pdf}'] = file.read().strip()
                    else:
                        pdf_percentages[f'{folder}/{subfolder}/{pdf}'] = '0'
    if 'your_notes' not in folders:
        os.makedirs(os.path.join(app.config['FOLDERS'], 'your_notes'), exist_ok=True)
    return render_template('index.html', folders=folders, pdf_files=pdf_files, subfolders=subfolders, pdf_percentages=pdf_percentages)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf_file' not in request.files:
        return redirect(request.url)
    file = request.files['pdf_file']
    folder = request.form.get('folder', '')
    subfolder = request.form.get('subfolder', '')
    if file and file.filename.endswith('.pdf'):
        filename = file.filename
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        if folder:
            if subfolder:
                folder_path = os.path.join(app.config['FOLDERS'], folder, subfolder)
            else:
                folder_path = os.path.join(app.config['FOLDERS'], folder)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(upload_path, os.path.join(folder_path, filename))
        else:
            default_folder = 'Default'
            folder_path = os.path.join(app.config['FOLDERS'], default_folder)
            os.makedirs(folder_path, exist_ok=True)
            shutil.move(upload_path, os.path.join(folder_path, filename))
    return redirect(url_for('index'))

@app.route('/move_pdf/<filename>/<source_folder>/<source_subfolder>', methods=['POST'])
def move_pdf(filename, source_folder, source_subfolder=None):
    target_folder = request.form.get('target_folder', source_folder)
    target_subfolder = request.form.get('target_subfolder', '')
    source_path = os.path.join(app.config['FOLDERS'], source_folder, source_subfolder if source_subfolder else '', filename)
    target_path = os.path.join(app.config['FOLDERS'], target_folder, target_subfolder if target_subfolder else '', filename)
    if not os.path.exists(source_path):
        return f"Source file not found: {source_path}", 404
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.move(source_path, target_path)
    except Exception as e:
        return f"Error moving file: {e}", 500
    return redirect(url_for('index'))

@app.route('/view_pdf/<folder>/<subfolder>/<filename>')
@app.route('/view_pdf/<folder>/<filename>')
def view_pdf(folder, subfolder=None, filename=None):
    folders = os.listdir(app.config['FOLDERS'])
    pdf_files = {}
    subfolders = {}
    if folder in folders:
        folder_path = os.path.join(app.config['FOLDERS'], folder)
        if os.path.isdir(folder_path):
            subfolders[folder] = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            pdf_files[folder] = {}
            for subfolder_name in subfolders[folder]:
                subfolder_path = os.path.join(folder_path, subfolder_name)
                pdf_files[folder][subfolder_name] = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f)) and f.endswith('.pdf')]
            pdf_files[folder][''] = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pdf')]
    return render_template('view_pdf.html', folder=folder, subfolder=subfolder, filename=filename, pdf_files=pdf_files, all_subfolders=subfolders)

@app.route('/serve_pdf/<folder>/<subfolder>/<filename>')
@app.route('/serve_pdf/<folder>/<filename>')
def serve_pdf(folder, subfolder=None, filename=None):
    if subfolder:
        file_path = os.path.join(app.config['FOLDERS'], folder, subfolder, filename)
    else:
        file_path = os.path.join(app.config['FOLDERS'], folder, filename)
    print(f"Serving PDF from: {file_path}")
    if os.path.exists(file_path):
        return send_from_directory(os.path.dirname(file_path), filename)
    else:
        return "PDF file not found.", 404

@app.route('/create_folder', methods=['POST'])
def create_folder():
    new_folder = request.form.get('new_folder')
    if new_folder:
        folder_path = os.path.join(app.config['FOLDERS'], new_folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    return redirect(url_for('index'))

@app.route('/create_subfolder', methods=['POST'])
def create_subfolder():
    folder = request.form.get('folder')
    new_subfolder = request.form.get('new_subfolder')
    if folder and new_subfolder:
        subfolder_path = os.path.join(app.config['FOLDERS'], folder, new_subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
    return redirect(url_for('index'))

@app.route('/update_pdf_scroll', methods=['POST'])
def update_pdf_scroll():
    filename = request.form.get('filename')
    folder = request.form.get('folder', '')
    subfolder = request.form.get('subfolder', '')
    scroll_percentage = request.form.get('scroll_percentage', '0')
    pdf_info_path = os.path.join('data', f'{folder}_{subfolder}_{filename}.txt')
    with open(pdf_info_path, 'w') as file:
        file.write(scroll_percentage)
    return '', 204

@app.route('/get_query_history/<folder>/<subfolder>/<filename>')
@app.route('/get_query_history/<folder>/<filename>')
def get_query_history(folder, subfolder=None, filename=None):
    pdf_id = f"{folder}_{subfolder}_{filename}" if subfolder else f"{folder}_{filename}"
    Query_db = Query()
    queries = db.search(Query_db.pdf_id == pdf_id)
    for query in queries:
        query['doc_id'] = query.doc_id
    return jsonify(queries)

@app.route('/delete_query', methods=['POST'])
def delete_query():
    data = request.json
    query_id = data.get('query_id')
    if query_id:
        db.remove(doc_ids=[query_id])
        return jsonify({'success': True})
    return jsonify({'success': False}), 400

@app.route('/create_note', methods=['POST'])
def create_note():
    title = request.form.get('title')
    content = request.form.get('note_content')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt=title, ln=1, align='C')
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=content)
    notes_folder = os.path.join(app.config['FOLDERS'], 'your_notes')
    os.makedirs(notes_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{title}_{timestamp}.pdf"
    pdf_path = os.path.join(notes_folder, filename)
    pdf.output(pdf_path)
    return redirect(url_for('index'))

@app.route('/delete_pdf/<folder>/<filename>')
@app.route('/delete_pdf/<folder>/<subfolder>/<filename>')
def delete_pdf(folder, subfolder=None, filename=None):
    if subfolder:
        file_path = os.path.join(app.config['FOLDERS'], folder, subfolder, filename)
    else:
        file_path = os.path.join(app.config['FOLDERS'], folder, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        pdf_info_path = os.path.join('data', f'{folder}_{subfolder}_{filename}.txt')
        if os.path.exists(pdf_info_path):
            os.remove(pdf_info_path)
    return redirect(url_for('index'))

@app.route('/chat')
def chat():
    chats = get_chat_list()
    chat_id_from_query = request.args.get('chat_id')
    current_chat_id = chat_id_from_query or session.get('current_chat_id')
    if not current_chat_id or not os.path.exists(get_conversation_file(current_chat_id)):
        current_chat_id = str(uuid.uuid4())
        print(f"Created new chat_id for index: {current_chat_id}")
    session['current_chat_id'] = current_chat_id
    conversation = load_conversation_history(current_chat_id)
    messages = conversation.get("messages", [])
    return render_template('chat.html',
                         chats=chats,
                         current_chat_id=current_chat_id,
                         mode=session.get('mode', 'user'))

@app.route('/api/chat/<chat_id>', methods=['POST'])
def chat_endpoint(chat_id):
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    data = request.json
    query = data.get('query')
    mode = data.get('mode', 'user')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    print(f"Processing chat request for chat_id: {chat_id} in mode: {mode}")
    session['current_chat_id'] = chat_id
    session['mode'] = mode
    try:
        response = process_query(chat_id, query, mode)
        return jsonify({
            "response": response,
            "chat_id": chat_id
        })
    except Exception as e:
        print(f"Error processing query for chat {chat_id}: {e}")
        return jsonify({"error": "Failed to process query"}), 500

@app.route('/api/chats', methods=['GET'])
def get_chats():
    chats = get_chat_list()
    return jsonify(chats)

@app.route('/api/chats/new', methods=['POST'])
def create_chat():
    chat_id = str(uuid.uuid4())
    print(f"Creating new chat with ID: {chat_id}")
    new_conversation = {
        "chat_id": chat_id,
        "created_at": datetime.now().isoformat(),
        "messages": []
    }
    save_conversation_history(chat_id, new_conversation)
    session['current_chat_id'] = chat_id
    session['mode'] = 'user'
    return jsonify({"chat_id": chat_id})

@app.route('/api/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    print(f"Loading chat: {chat_id}")
    conversation = load_conversation_history(chat_id)
    messages = conversation.get("messages", [])
    session['current_chat_id'] = chat_id
    return jsonify({
        "messages": messages,
        "chat_id": chat_id
    })

@app.route('/api/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    try:
        history_file = get_conversation_file(chat_id)
        if os.path.exists(history_file):
            os.remove(history_file)
            print(f"Deleted chat file: {history_file}")
        for filename in [f"final_query_{chat_id}.txt", f"content_{chat_id}.txt"]:
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Deleted associated file: {filename}")
        if session.get('current_chat_id') == chat_id:
            session.pop('current_chat_id', None)
        return jsonify({"success": True})
    except Exception as e:
        print(f"Error deleting chat {chat_id}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/chats/<chat_id>/debug', methods=['GET'])
def get_chat_debug(chat_id):
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400
    conversation = load_conversation_history(chat_id)
    return jsonify(conversation)


@app.route('/query_rag', methods=['POST'])
def query_rag():
    data = request.json
    query = data.get('query')
    filename = data.get('filename')
    folder = data.get('folder')
    subfolder = data.get('subfolder', '') or None
    if subfolder == 'None':
        subfolder = None
    llm_choice = data.get('llm_choice', 'groq')
    if subfolder:
        pdf_path = os.path.join(app.config['FOLDERS'], folder, subfolder, filename)
    else:
        pdf_path = os.path.join(app.config['FOLDERS'], folder, filename)
    print(f"Constructed PDF path: {pdf_path}")
    if not os.path.exists(pdf_path):
        return jsonify({'response': f'PDF file not found at {pdf_path}.'}), 404
    qa_chain = load_knowledge_base(pdf_path, llm_choice)
    response = qa_chain.invoke(input=query)
    pdf_id = f"{folder}_{subfolder}_{filename}" if subfolder else f"{folder}_{filename}"
    db.insert({
        'pdf_id': pdf_id,
        'query': query,
        'response': response['result'],
        'timestamp': datetime.now().isoformat()
    })
    return jsonify({'response': response['result']})

@app.route('/stream_rag', methods=['POST'])
def stream_rag():
    data = request.json
    query = data.get('query')
    filename = data.get('filename')
    folder = data.get('folder')
    subfolder = data.get('subfolder', '') or None
    if subfolder == 'None':
        subfolder = None
    llm_choice = data.get('llm_choice', 'groq')
    if subfolder:
        pdf_path = os.path.join(app.config['FOLDERS'], folder, subfolder, filename)
    else:
        pdf_path = os.path.join(app.config['FOLDERS'], folder, filename)
    if not os.path.exists(pdf_path):
        return jsonify({'response': f'PDF file not found at {pdf_path}.'}), 404
    qa_chain = load_knowledge_base(pdf_path, llm_choice)
    def generate():
        full_response = ""
        for chunk in qa_chain.run(query):
            full_response += chunk
            yield chunk
        pdf_id = f"{folder}_{subfolder}_{filename}" if subfolder else f"{folder}_{filename}"
        db.insert({
            'pdf_id': pdf_id,
            'query': query,
            'response': full_response,
            'timestamp': datetime.now().isoformat()
        })
    return Response(generate(), mimetype='text/plain')

if __name__ == '__main__':
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("Please set the GROQ_API_KEY in the .env file before running the application.")
    app.run(host='0.0.0.0', port=8080, debug=True)

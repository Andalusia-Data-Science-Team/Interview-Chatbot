import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import os
import json
import re
import requests
import secrets
import sqlite3
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration (unchanged) ---
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY", "")
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
MODEL_NAME = "accounts/fireworks/models/llama-v3p3-70b-instruct"

# Email config
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER", "your-email@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "your-app-password")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "your-email@gmail.com")

# Admin password (change this!)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

MAX_QUESTIONS = 4
OUTPUT_DIR = "results"
DB_PATH = "interviews.db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------
# Database Setup & Functions
# --------------------------
def init_database():
    """Initializes the interviews table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS interviews
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 candidate_name TEXT NOT NULL,
                 candidate_email TEXT,
                 token TEXT UNIQUE NOT NULL,
                 status TEXT DEFAULT 'pending',
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 started_at TIMESTAMP,
                 completed_at TIMESTAMP,
                 expires_at TIMESTAMP,
                 ip_address TEXT,
                 result_filepath TEXT,
                 evaluation_json_data TEXT,
                 average_score REAL,   -- NEW column for storing the final score (e.g., 7.5)
                 recommendation TEXT   -- NEW column for storing the final recommendation (e.g., 'Hire')
                )''')
    conn.commit()
    conn.close()


def add_new_columns_if_not_exists():
    """Ensures new schema columns exist in an existing database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check for result_filepath
    try:
        c.execute("SELECT result_filepath FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN result_filepath TEXT")
        conn.commit()

    # Check for evaluation_json_data
    try:
        c.execute("SELECT evaluation_json_data FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN evaluation_json_data TEXT")
        conn.commit()
        
    # Check for average_score
    try:
        c.execute("SELECT average_score FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN average_score REAL")
        conn.commit()
        
    # Check for recommendation
    try:
        c.execute("SELECT recommendation FROM interviews LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("ALTER TABLE interviews ADD COLUMN recommendation TEXT")
        conn.commit()
    
    conn.close()

# Run initialization and schema update
init_database()
add_new_columns_if_not_exists()

def reset_database():
    """Deletes all records from the interviews table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM interviews')
    conn.commit()
    conn.close()
    return True

def generate_interview_token(candidate_name, candidate_email="", hours_valid=48):
    """Generate unique token and store in database"""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=hours_valid)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO interviews (candidate_name, candidate_email, token, expires_at)
                 VALUES (?, ?, ?, ?)''',
              (candidate_name, candidate_email, token, expires_at))
    conn.commit()
    conn.close()
    
    return token

def validate_token(token):
    """Check if token is valid and not used"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT id, candidate_name, status, expires_at, completed_at 
                 FROM interviews WHERE token = ?''', (token,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return {"valid": False, "message": "❌ Invalid interview link"}
    
    interview_id, candidate_name, status, expires_at, completed_at = result
    
    if status == 'completed' or completed_at:
        return {"valid": False, "message": "❌ This interview has already been completed"}
    
    expires_dt = datetime.fromisoformat(expires_at)
    if datetime.now() > expires_dt:
        return {"valid": False, "message": "❌ This interview link has expired"}
    
    return {"valid": True, "interview_id": interview_id, "candidate_name": candidate_name}

def mark_interview_started(token):
    """Mark interview as started"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE interviews SET started_at = ?, status = 'in_progress'
                 WHERE token = ?''', (datetime.now(), token))
    conn.commit()
    conn.close()

def mark_interview_completed(token, filepath, evaluation_json_dump, evaluation_dict):
    """
    Mark interview as completed and save result metadata, including final score and recommendation.
    We now accept the parsed dictionary (evaluation_dict) to easily extract metrics.
    """
    
    avg_score = None
    recommendation = None
    if "final_evaluation" in evaluation_dict:
        # We need to handle cases where 'average_score' might be a string (e.g., 'N/A')
        score_str = evaluation_dict["final_evaluation"].get("average_score")
        try:
            avg_score = float(score_str)
        except (ValueError, TypeError):
            # If it can't be converted to float (e.g., 'N/A' or None), keep it as None
            avg_score = None
            
        recommendation = evaluation_dict["final_evaluation"].get("recommendation")
        
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE interviews SET completed_at = ?, 
                                       status = 'completed',
                                       result_filepath = ?,
                                       evaluation_json_data = ?,
                                       average_score = ?,
                                       recommendation = ?
                 WHERE token = ?''', 
                 (datetime.now(), filepath, evaluation_json_dump, avg_score, recommendation, token))
    conn.commit()
    conn.close()

def get_all_interviews():
    """Get all interviews for admin panel, including result path, average score, and recommendation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT candidate_name, candidate_email, token, status, 
                      created_at, completed_at, expires_at, result_filepath, evaluation_json_data,
                      average_score, recommendation 
                 FROM interviews ORDER BY created_at DESC''')
    # Returns (name, email, token, status, created, completed, expires, filepath, json_data, avg_score, recommendation)
    results = c.fetchall()
    conn.close()
    return results

# --------------------------
# Email Notification (updated error handling)
# --------------------------
def send_email_notification(candidate_name, filename):
    """Send email when interview is completed with better error handling."""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"Interview Completed - {candidate_name}"
        
        body = f"""
Interview completed for: {candidate_name}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Results file: {filename}

Check the results folder or the Admin Panel for detailed evaluation.
"""
        msg.attach(MIMEText(body, 'plain'))
        
        # Connection with timeout
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) # Added 10-second timeout
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except smtplib.SMTPAuthenticationError:
        print(f"Email error: Failed to log in. Check EMAIL_USER and EMAIL_PASSWORD (App Password needed for Gmail).")
        return False
    except smtplib.SMTPConnectError as e:
        print(f"Email error: Could not connect to SMTP server. Check server address, port, and network firewall. Details: {e}")
        return False
    except TimeoutError as e:
        print(f"Email error: Connection timed out. Check network connectivity or firewall. Details: {e}")
        return False
    except Exception as e:
        print(f"Email error: An unexpected error occurred. Details: {e}")
        return False

# --------------------------
# Prompts & Fireworks API Functions (unchanged in logic)
# --------------------------
EVALUATOR_PROMPT = """
You are a medical technical interviewer and evaluator specialized in Dentistry.

- You will receive candidate answers for 4 interview questions.
- For each answer, provide:
  Score (0–10), Strengths, Weaknesses, Suggestions for improvement.
- After all answers, provide a final overall evaluation:
  Average Score, Overall Strengths, Overall Weaknesses, and Final Recommendation (Hire / Consider / Reject).

The output must be valid JSON in the format:

{
  "evaluations": {
    "Question 1": {"score": 8, "strengths": "...", "weaknesses": "...", "suggestions": "..."},
    "Question 2": {...},
    "Question 3": {...},
    "Question 4": {...}
  },
  "final_evaluation": {
    "average_score": 7.5,
    "overall_strengths": "...",
    "overall_weaknesses": "...",
    "recommendation": "Consider"
  }
}
Return only valid JSON, without any explanations or markdown fences.
"""

FALLBACK_QUESTIONS = [
    "Describe your experience with different types of dental materials and their respective properties (e.g., strength, biocompatibility, aesthetics).",
    "Tell me about a time you had to deal with a difficult patient. How did you handle the situation?",
    "Describe a complex dental procedure you performed and how you ensured patient safety and comfort.",
    "How do you stay updated with the latest advancements in dental technologies and treatments?"
]

def call_fireworks_api(prompt, system_prompt="You are a helpful assistant."):
    # ... (Implementation remains the same)
    if not FIREWORKS_API_KEY:
        print("API Key missing. Cannot call Fireworks.")
        return None

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }

    try:
        response = requests.post(FIREWORKS_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Fireworks API Error: {e}")
        # Return structured error for graceful failure
        return json.dumps({"error": f"Fireworks API Error: {e}. Check API key and configuration."})

def clean_markdown_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^```[a-zA-Z]*\n", "", s)
    s = re.sub(r"\n```$", "", s)
    return s.strip()

def generate_all_questions(num=MAX_QUESTIONS):
    # ... (Implementation remains the same)
    prompt = f"""
    Generate {num} technical dentistry interview questions.
    Return only valid JSON like:
    {{
      "questions": [
        "Question 1 text...",
        "Question 2 text...",
        "Question 3 text...",
        "Question 4 text..."
      ]
    }}
    """
    response = call_fireworks_api(prompt, "You are a technical interviewer specializing in dentistry.")
    if response:
        try:
            raw = clean_markdown_fences(response)
            data = json.loads(raw)
            return data.get("questions", FALLBACK_QUESTIONS[:num])
        except Exception as e:
            print(f"Failed to parse questions: {e}")
    return FALLBACK_QUESTIONS[:num]

def evaluate_with_fireworks(answers: dict) -> dict:
    # ... (Implementation remains the same)
    answers_text = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(answers.items())])
    full_prompt = EVALUATOR_PROMPT + "\n\nCandidate Answers:\n" + answers_text
    response = call_fireworks_api(full_prompt, "You are a medical technical interviewer.")
    
    if response:
        try:
            # Check if the response is an error JSON returned by call_fireworks_api
            try:
                error_check = json.loads(response)
                if "error" in error_check:
                    return error_check
            except json.JSONDecodeError:
                pass # Not an error JSON, continue parsing
            
            raw_text = clean_markdown_fences(response)
            return json.loads(raw_text)
        except Exception as e:
            print(f"Failed to parse evaluation JSON: {e}")
            return {"error": f"Evaluation failed (JSON parsing error): {e}"}
    else:
        return {"error": "No valid response from API or response was None."}

def save_results_txt(candidate_name, answers, evaluation_json):
    """Saves the comprehensive evaluation to a local TXT file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = candidate_name.replace(" ", "_") or "candidate"
    filename = f"{OUTPUT_DIR}/{safe_name}_{timestamp}.txt"
    filepath = str(Path(filename).resolve()) # Get absolute path for storage

    lines = [f"Candidate: {candidate_name}",
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "="*60, "", "Interview Answers", "-"*60]

    for i, (q, a) in enumerate(answers.items(), 1):
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a if a.strip() else '[No answer]'}")
        lines.append("")

    lines.append("Evaluation")
    lines.append("-"*60)
    
    if "error" in evaluation_json:
        lines.append(f"ERROR: Evaluation failed: {evaluation_json['error']}")
    else:
        if "evaluations" in evaluation_json:
            for q, details in evaluation_json["evaluations"].items():
                lines.append(q)
                lines.append(f"   Score: {details.get('score','N/A')}")
                lines.append(f"   Strengths: {details.get('strengths','N/A')}")
                lines.append(f"   Weaknesses: {details.get('weaknesses','N/A')}")
                lines.append(f"   Suggestions: {details.get('suggestions','N/A')}")
                lines.append("")

        if "final_evaluation" in evaluation_json:
            fe = evaluation_json["final_evaluation"]
            lines.append("Final Evaluation")
            lines.append("-"*60)
            lines.append(f"Average Score: {fe.get('average_score','N/A')}")
            lines.append(f"Overall Strengths: {fe.get('overall_strengths','N/A')}")
            lines.append(f"Overall Weaknesses: {fe.get('overall_weaknesses','N/A')}")
            lines.append(f"Recommendation: {fe.get('recommendation','N/A')}")

    content = "\n".join(lines)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
        
    return filepath, content

# --------------------------
# Dash App
# --------------------------
app = dash.Dash(__name__, 
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
                suppress_callback_exceptions=True) # Suppress for conditional layouts
app.title = "Dentistry Interview Evaluator"
server = app.server

# --------------------------
# Layouts
# --------------------------
def create_header():
    # ... (Header function remains the same)
    return dbc.Container([
        html.Div([
            html.H1([
                html.I(className="fas fa-tooth me-3"),
                "Dentistry Interview Evaluator"
            ], className="display-5 fw-bold mb-0", style={"color": "#B68648"}),
            html.P("Alandalusia Health Egypt",
                   className="lead mb-0 opacity-75",
                   style={"color": "#B68648"})
        ], style={
            "backgroundColor": "#F2F2F2",
            "padding": "20px",
            "textAlign": "center",
            "borderRadius": "0 0 20px 20px"
        })
    ], fluid=True, className="p-0 mb-4")

def admin_panel_layout():
    """Admin panel for generating interview links and viewing results"""
    return dbc.Container([
        create_header(),
        dbc.Card([
            dbc.CardHeader(html.H3("🔐 Admin Panel")),
            dbc.CardBody([
                html.H5("Generate New Interview Link"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Candidate Name *"),
                        dbc.Input(id="admin-candidate-name", placeholder="Enter name")
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Candidate Email (optional)"),
                        dbc.Input(id="admin-candidate-email", placeholder="Enter email", type="email")
                    ], width=6)
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Link Valid For (hours)"),
                        dbc.Input(id="admin-hours-valid", value="48", type="number")
                    ], width=4)
                ]),
                dbc.Button("Generate Link", id="generate-link-btn", color="primary", className="mt-3"),
                html.Div(id="generated-link-output", className="mt-3"),
                html.Hr(),
                html.H5("Database Management"),
                dbc.Button("⚠️ Reset Database (Delete All Data)", id="reset-db-btn", color="danger", className="mt-2 mb-3"),
                html.Div(id="db-reset-output", className="mb-3"), # Output for reset feedback
                html.Hr(),
                html.H5("All Interviews & Results"),
                html.Div(id="interviews-list"),
                dbc.Button("Refresh List", id="refresh-list-btn", color="secondary", className="mt-2"),
                # Component to trigger server-side file download
                dcc.Download(id="download-result-file")
            ])
        ])
    ], fluid=True, className="p-4")

def interview_layout():
    """Candidate interview interface (no result components here)"""
    return dbc.Container([
        create_header(),
        # --- Persistent components for callbacks ---
        dcc.Store(id='questions-store', data=[]),
        dcc.Store(id='answers-store', data={}),
        dcc.Store(id='current-index-store', data=0),
        dcc.Store(id='step-store', data=0),
        dcc.Store(id='token-store', data=''),
        dcc.Store(id='candidate-store', data=''),
        dcc.Store(id='evaluation-store', data=None), 
        html.Div(id="token-validation-result"),
        # --- Permanent element for feedback, now globally available in the layout ---
        html.Div(id="submit-feedback", className="mt-2"), 
        # ------------------------------------------
        html.Div(id="interview-section"),
        html.Div(id="evaluation-section") # Only shows completion message / loading spinner
    ], fluid=True, className="p-4")

# --------------------------
# Main Layout & Routing (unchanged)
# --------------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname'),
     Input('url', 'search')]
)
def display_page(pathname, search):
    # Admin panel
    if pathname == '/admin':
        if search and f"?pw={ADMIN_PASSWORD}" in search:
            return admin_panel_layout()
        else:
            return dbc.Container([
                create_header(),
                dbc.Alert("Access Denied. Use correct admin URL.", color="danger")
            ])
    
    # Interview page with token
    elif pathname and pathname.startswith('/interview/'):
        token = pathname.split('/interview/')[-1]
        return interview_layout()
    
    # Default home
    else:
        return dbc.Container([
            create_header(),
            dbc.Card([
                dbc.CardBody([
                    html.H3("Welcome to Dentistry Interview System"),
                    html.P("If you received an interview link, please use that link to start."),
                    html.P("Administrators can access the admin panel to generate interview links.")
                ])
            ])
        ], className="p-4")

# --------------------------
# Admin Panel Callbacks
# --------------------------
@app.callback(
    Output('db-reset-output', 'children'),
    [Input('reset-db-btn', 'n_clicks')]
)
def handle_db_reset(n_clicks):
    """Resets the database and confirms action."""
    if n_clicks and n_clicks > 0:
        reset_database()
        return dbc.Alert("✅ Database cleared successfully! All interview data has been removed.", color="success")
    return ""

@app.callback(
    Output('generated-link-output', 'children'),
    [Input('generate-link-btn', 'n_clicks')],
    [State('admin-candidate-name', 'value'),
     State('admin-candidate-email', 'value'),
     State('admin-hours-valid', 'value'),
     State('url', 'href')]
)
def generate_link(n_clicks, name, email, hours, current_url):
    # ... (Link generation remains the same)
    if not n_clicks or not name:
        return ""
    
    hours_valid = int(hours) if hours else 48
    token = generate_interview_token(name, email or "", hours_valid)
    
    # Build full URL
    base_url = current_url.split('/admin')[0] if '/admin' in current_url else current_url.rstrip('/')
    interview_url = f"{base_url}/interview/{token}"
    
    return dbc.Alert([
        html.H5("✅ Interview Link Generated!", className="mb-3"),
        html.P(f"Candidate: {name}"),
        html.P(f"Valid for: {hours_valid} hours"),
        html.Hr(),
        html.Label("Copy this link and send via WhatsApp:"),
        dbc.InputGroup([
            dbc.Input(value=interview_url, id="link-to-copy", readonly=True),
            # Note: Copy functionality requires client-side JavaScript, which Dash can't easily do natively. 
            # The input is provided for easy manual copy.
        ]),
        html.Small("Expires: " + (datetime.now() + timedelta(hours=hours_valid)).strftime('%Y-%m-%d %H:%M'), 
                   className="text-muted mt-2 d-block")
    ], color="success")

@app.callback(
    Output('interviews-list', 'children'),
    [Input('refresh-list-btn', 'n_clicks'),
     Input('generate-link-btn', 'n_clicks'),
     Input('db-reset-output', 'children')] # Trigger refresh after reset
)
def update_interviews_list(refresh_clicks, generate_clicks, reset_output):
    interviews = get_all_interviews()
    
    if not interviews:
        return dbc.Alert("No interviews yet", color="info")
    
    table_rows = []
    for interview in interviews:
        # Unpack the 11 columns from the new SELECT query
        name, email, token, status, created, completed, expires, filepath, json_data, avg_score, recommendation = interview
        
        status_badge = {
            'pending': dbc.Badge("Pending", color="warning"),
            'in_progress': dbc.Badge("In Progress", color="info"),
            'completed': dbc.Badge("Completed", color="success")
        }.get(status, dbc.Badge(status, color="secondary"))
        
        # Format the score for display
        score_display = f"{avg_score:.1f}" if avg_score is not None else "-"
        
        # Download button is only available if status is completed AND a filepath exists
        download_col = html.Td("-")
        if status == 'completed' and filepath:
            download_col = html.Td(dbc.Button(
                "Download TXT", 
                id={"type": "download-btn", "index": token}, # Dynamic ID for download
                color="info", 
                size="sm"
            ))

        table_rows.append(html.Tr([
            html.Td(name),
            html.Td(email or "-"),
            html.Td(status_badge),
            html.Td(completed[:16] if completed else "-"),
            html.Td(score_display), # NEW: Average Score
            html.Td(recommendation or "-"), # NEW: Recommendation
            download_col
        ]))
    
    # Update the table headers to include the new columns
    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Name"),
            html.Th("Email"),
            html.Th("Status"),
            html.Th("Completed"),
            html.Th("Avg Score"),
            html.Th("Recommendation"),
            html.Th("Results")
        ])),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, striped=True, size="sm")

@app.callback(
    Output('download-result-file', 'data'),
    [Input({'type': 'download-btn', 'index': dash.ALL}, 'n_clicks')]
)
def download_admin_result(n_clicks):
    """Callback to handle dynamic download requests from the admin table."""
    trigger = callback_context.triggered
    if not trigger or not any(n_clicks):
        return dash.no_update

    # Find the button that was clicked
    button_id = trigger[0]['prop_id'].split('.')[0]
    
    try:
        # Extract the token from the dynamic ID
        token = json.loads(button_id)['index']
    except Exception:
        print("Error parsing download button ID.")
        return dash.no_update

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Fetch filepath and candidate name using the token
    c.execute('''SELECT candidate_name, result_filepath FROM interviews WHERE token = ?''', (token,))
    result = c.fetchone()
    conn.close()

    if result:
        candidate_name, filepath = result
        try:
            # Read the file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use the saved filename for download
            filename = os.path.basename(filepath)
            
            return dcc.send_string(content, filename=filename)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            # If file is missing, return a dummy text file with error
            return dcc.send_string(f"Error: Could not find or read file at path: {filepath}", filename="error_download.txt")

    return dash.no_update

# --------------------------
# Interview Callbacks (Candidate View)
# --------------------------
@app.callback(
    [Output('token-store', 'data'),
     Output('candidate-store', 'data'),
     Output('token-validation-result', 'children'),
     Output('questions-store', 'data'),
     Output('step-store', 'data')],
    [Input('url', 'pathname')]
)
def validate_interview_token(pathname):
    # ... (Token validation and question generation remains the same)
    if not pathname or not pathname.startswith('/interview/'):
        return '', '', '', [], 0
    
    token = pathname.split('/interview/')[-1]
    validation = validate_token(token)
    
    if not validation['valid']:
        return '', '', dbc.Alert(validation['message'], color="danger"), [], 0
    
    # Valid token - mark as started and generate questions
    mark_interview_started(token)
    questions = generate_all_questions(MAX_QUESTIONS)
    
    welcome_msg = dbc.Alert([
        html.H4(f"Welcome, {validation['candidate_name']}! 👋"),
        html.P("Your interview will begin shortly. Please answer all questions thoughtfully."),
        html.P("⚠️ Note: This link can only be used once. Do not refresh the page.")
    ], color="info")
    
    return token, validation['candidate_name'], welcome_msg, questions, 1

@app.callback(
    [Output('interview-section', 'children'),
     Output('evaluation-section', 'children')],
    [Input('questions-store', 'data'),
     Input('current-index-store', 'data'),
     Input('answers-store', 'data'),
     Input('step-store', 'data')]
)
def update_interview_display(questions, current_idx, answers, step):
    """Updates candidate interview display based on the current step."""
    interview_content, evaluation_content = [], []

    if step == 1 and questions:
        # In-progress section (unchanged)
        if current_idx < len(questions):
            q_text = questions[current_idx]
            interview_content = [
                dbc.Card([
                    dbc.CardHeader(html.H4(f"Question {current_idx + 1}/{MAX_QUESTIONS}")),
                    dbc.CardBody([
                        html.P(q_text, className="lead mb-3"),
                        # Note: submit-feedback is now rendered permanently in interview_layout
                        dbc.Textarea(id="answer-input", placeholder="Type your answer here...",
                                     style={"height": "200px"}, className="mb-3",
                                     value=answers.get(q_text, "")),
                        dbc.Button("Submit Answer", id="submit-answer-btn", color="success", className="me-2"),
                    ])
                ], className="mb-3"),
                dbc.Progress(value=(len(answers)/MAX_QUESTIONS)*100, className="mb-2"),
                html.P(f"Progress: {len(answers)} / {MAX_QUESTIONS} answered", className="text-muted")
            ]
    elif step == 2:
        # Completion view (only shows loading spinner while evaluation runs)
        evaluation_content = [
            dbc.Alert("✅ Interview completed! Evaluation is running...", color="success"),
            html.Div(id="evaluation-display"),
        ]
        
    return interview_content, evaluation_content

@app.callback(
    [Output('evaluation-store', 'data', allow_duplicate=True),
     Output('submit-feedback', 'children', allow_duplicate=True)],
    [Input('step-store', 'data')],
    [State('answers-store', 'data'),
     State('evaluation-store', 'data'),
     State('token-store', 'data'),
     State('candidate-store', 'data')],
    prevent_initial_call=True
)
def trigger_evaluation(step, answers, evaluation, token, candidate_name):
    """Triggers the AI evaluation and saves the results to the database."""
    if step == 2 and evaluation is None and answers:
        
        # 1. Evaluate
        eval_result = evaluate_with_fireworks(answers)
        evaluation_json_dump = json.dumps(eval_result, ensure_ascii=False)
        
        # 2. Save results locally (TXT file)
        filepath, content = save_results_txt(candidate_name, answers, eval_result)
        
        # 3. Mark interview as completed and save metadata (filepath, JSON, Avg Score, Recommendation)
        # Note: Passing the parsed dict (eval_result) to the function now.
        mark_interview_completed(token, filepath, evaluation_json_dump, eval_result)
        
        # 4. Send email notification
        send_email_notification(candidate_name, os.path.basename(filepath))
        
        # Return the evaluation data to the store
        return eval_result, ""
    
    # If step is not 2, do nothing with evaluation or feedback
    return dash.no_update, dash.no_update

@app.callback(Output('evaluation-display', 'children'),
              [Input('evaluation-store', 'data')])
def display_evaluation(evaluation):
    """Displays only a confirmation message to the candidate."""
    if not evaluation:
        return dbc.Spinner(color="primary")
        
    if "error" in evaluation:
        # Candidate sees a generic failure message, print details to console for admin
        print(f"Candidate Evaluation Error Details: {evaluation['error']}")
        return dbc.Alert("❌ An error occurred during evaluation. The administrator has been notified.", color="danger")
        
    # Success message for the candidate
    return dbc.Alert("✅ Thank you. Your interview is complete. Results have been sent to the administrator.", color="success")


@app.callback(
    [Output('answers-store', 'data', allow_duplicate=True),
     Output('current-index-store', 'data', allow_duplicate=True),
     Output('step-store', 'data', allow_duplicate=True),
     Output('submit-feedback', 'children', allow_duplicate=True)],
    [Input('submit-answer-btn', 'n_clicks')],
    [State('answer-input', 'value'),
     State('questions-store', 'data'),
     State('current-index-store', 'data'),
     State('answers-store', 'data')],
    prevent_initial_call=True
)
def submit_answer(n_clicks, answer_text, questions, current_idx, answers):
    if not n_clicks:
        return answers, current_idx, dash.no_update, dash.no_update
        
    if not answer_text or not answer_text.strip():
        return answers, current_idx, dash.no_update, dbc.Alert("⚠️ Please write an answer before submitting.", color="warning")

    q_text = questions[current_idx]
    answers[q_text] = answer_text.strip()
    
    if current_idx >= MAX_QUESTIONS - 1:
        # Last question submitted, move to evaluation (Step 2)
        return answers, current_idx, 2, dbc.Alert("Answer submitted ✅ Starting evaluation...", color="success")
    else:
        # Move to next question (Step 1)
        return answers, current_idx + 1, 1, dbc.Alert("Answer submitted ✅ Loading next question...", color="success")

# --------------------------
# Run Server
# --------------------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8051)

# HR Interview Evaluator

An AI-powered HR interview system that generates tailored interview questions, evaluates candidate responses, and provides automated scoring and recommendations.

## Features

- **AI-Powered Question Generation**: Creates customized interview questions based on candidate profiles using OpenRouter API (DeepSeek model)
- **Automated Evaluation**: Scores candidate answers (0-10) and provides strengths/weaknesses analysis
- **Hiring Recommendations**: Generates Hire/Consider/Reject recommendations based on overall performance
- **Admin Dashboard**: Generate interview links, track candidate progress, and download results
- **Email Notifications**: Sends alerts to admin when interviews are completed
- **Secure Access**: Token-based interview links with expiration times
- **SQLite Database**: Stores all interview data and results locally

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this project

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables (optional - defaults are provided in `config.py`):
```bash
cp .env.example .env
```

## Configuration

Edit `config.py` or create a `.env` file with the following settings:

### OpenRouter API (Required)
```python
OPENROUTER_API_KEY = "your-openrouter-api-key"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "deepseek/deepseek-chat"
```

### Email Settings (Optional)
```python
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "your-email@gmail.com"
EMAIL_PASSWORD = "your-gmail-app-password"
ADMIN_EMAIL = "admin-email@gmail.com"
```

**Note:** For Gmail, you need to create an App Password:
1. Go to Google Account Settings → Security
2. Enable 2-Step Verification
3. Search for "App Passwords"
4. Create a new app password and use it here

### Admin Settings
```python
ADMIN_PASSWORD = "admin123"  # Change this for production
NUM_QUESTIONS = 5  # Number of interview questions per candidate
```

## Running the Application

### Development
```bash
python app.py
```

The server will start on `http://127.0.0.1:8051` and `http://0.0.0.0:8051`

### Production
```bash
gunicorn -w 4 -b 0.0.0.0:8051 app:app
```

## Usage

### Admin Panel

1. Access the admin panel:
   ```
   http://localhost:8051/admin?pw=your-admin-password
   ```

2. **Generate Interview Link**:
   - Enter candidate name
   - Enter candidate email (optional)
   - Set link validity period (default: 48 hours)
   - Click "Generate Link"
   - Share the generated link with the candidate

3. **View Interviews**:
   - See all interviews with status (pending/in_progress/completed)
   - View scores and recommendations
   - Download completed interview results as text files

### Candidate Interview Flow

1. Candidate opens the interview link
2. **Profile Stage**: Fill in personal details:
   - Full Name
   - Age
   - Years of Experience
   - Location
   - Notice Period
   - Expected Salary
   - Specialist/Role

3. **Question Stage**: Answer AI-generated interview questions
4. **Completion**: Receive confirmation, results are sent to admin

## Project Structure

```
.
├── app.py              # Flask application and routes
├── config.py           # Configuration settings
├── db.py               # SQLite database operations
├── llm.py              # OpenRouter API integration for AI features
├── mailer.py           # Email notification functionality
├── results.py          # Result file generation
├── templates.py        # HTML templates and styling
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (create this)
├── interviews.db       # SQLite database (auto-created)
└── results/            # Directory for interview result files
```

## API Endpoints

- `GET /` - Home page
- `GET/POST /admin?pw=password` - Admin dashboard
- `GET /admin/download/<token>?pw=password` - Download interview results
- `GET/POST /interview/<token>` - Candidate interview interface

## Database Schema

The `interviews.db` SQLite database contains:
- `token` - Unique interview identifier
- `name` - Candidate name
- `email` - Candidate email
- `status` - pending/in_progress/completed
- `created_at` - Interview creation timestamp
- `expires_at` - Link expiration timestamp
- `profile` - JSON-encoded candidate profile
- `questions` - JSON-encoded interview questions
- `answers` - JSON-encoded candidate answers
- `result_path` - Path to result file
- `score` - Average score (0-10)
- `recommendation` - Hire/Consider/Reject

## Security Notes

- Change the default `ADMIN_PASSWORD` in production
- Use environment variables for sensitive credentials
- Interview links expire after the configured time period
- Each interview link can only be used once
- Admin routes require password authentication

## Troubleshooting

**"Access denied" on admin panel:**
- Verify the password parameter: `?pw=admin123`
- Check `ADMIN_PASSWORD` in config.py or .env file
- Restart the server after changing configuration

**Email not sending:**
- Verify Gmail App Password (not regular password)
- Check SMTP settings in config.py
- Ensure 2-Step Verification is enabled on Google account

**AI not generating questions:**
- Verify OpenRouter API key is valid
- Check OPENROUTER_MODEL setting
- Ensure you have API credits available

## License

This project is provided as-is for HR interview automation purposes.

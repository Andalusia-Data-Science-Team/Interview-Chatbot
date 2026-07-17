import json
import re
import requests
import config

FALLBACK_QUESTIONS = [
    "Tell me about yourself and your professional background.",
    "What are your key strengths for this role?",
    "What is a weakness you're working to improve?",
    "Describe a challenge you faced at work and how you handled it.",
    "Why should we hire you?",
]


def _call(prompt, system):
    if not config.OPENROUTER_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {config.OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": config.OPENROUTER_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    try:
        r = requests.post(config.OPENROUTER_BASE_URL, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print("OpenRouter error:", e)
        return None


def _clean_json(s):
    s = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", s.strip())
    return json.loads(s)


def generate_questions(profile):
    n = config.NUM_QUESTIONS
    prompt = f"""Generate {n} HR interview questions tailored to this candidate:
{json.dumps(profile)}
Include one question about weaknesses or areas for improvement.
Return only JSON: {{"questions": ["...", ...]}}"""
    raw = _call(prompt, "You are an expert HR interviewer across all industries.")
    if raw:
        try:
            qs = _clean_json(raw).get("questions", [])
            if len(qs) == n:
                return qs
        except Exception as e:
            print("Question parse error:", e)
    return FALLBACK_QUESTIONS[:n]


def evaluate(profile, answers):
    prompt = f"""You are an HR evaluator. Candidate profile: {json.dumps(profile)}
Answers: {json.dumps(answers)}
For each question give a score (0-10), strengths, and weaknesses.
Then give a final_evaluation with: average_score, overall_strengths, overall_weaknesses,
recommendation (Hire / Consider / Reject).
Return only JSON: {{"evaluations": {{...}}, "final_evaluation": {{...}}}}"""
    raw = _call(prompt, "You are a strict but fair HR evaluator.")
    if raw:
        try:
            return _clean_json(raw)
        except Exception as e:
            return {"error": f"Could not parse evaluation: {e}"}
    return {"error": "No response from evaluator (missing API key or API error)."}

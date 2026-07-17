import json
import secrets
from datetime import datetime, timedelta

from flask import Flask, request, send_file, abort

import config
import db
import llm
import results
from mailer import notify_admin
from templates import page, alert, esc

app = Flask(__name__)


@app.route("/")
def home():
    body = ('<div class="card"><p>If you received an interview link, use it to start.</p>'
            '<p>Admins: visit <code>/admin?pw=your-password</code></p></div>')
    return page("HR Interview Evaluator", body)


# ---------------- Admin ----------------
@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.args.get("pw") != config.ADMIN_PASSWORD:
        return page("Admin", alert("Access denied.", "error"))

    msg = ""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        hours = int(request.form.get("hours") or 48)
        if name:
            token = secrets.token_urlsafe(16)
            expires = (datetime.now() + timedelta(hours=hours)).isoformat()
            db.create_interview(token, name, email, expires)
            link = request.host_url + f"interview/{token}"
            msg = alert(f'Link for <b>{esc(name)}</b>:<br>'
                        f'<input value="{link}" readonly onclick="this.select()">', "success")

    rows = db.all_interviews()
    table_rows = "".join(
        f'<tr><td>{esc(r["name"])}</td>'
        f'<td><span class="badge {r["status"]}">{r["status"]}</span></td>'
        f'<td class="score">{r["score"] or "-"}</td>'
        f'<td>{esc(r["recommendation"] or "-")}</td>'
        f'<td>{f"<a class=btn href=/admin/download/{r['token']}?pw={config.ADMIN_PASSWORD}>Download</a>" if r["status"] == "completed" else "-"}</td>'
        "</tr>"
        for r in rows
    )
    table = f"<tr><th>Name</th><th>Status</th><th>Score</th><th>Recommendation</th><th></th></tr>{table_rows}"

    body = f"""
    {msg}
    <div class="card">
      <h3>Generate Interview Link</h3>
      <form method="post">
        <div class="field"><label>Candidate Name</label><input name="name" required></div>
        <div class="field"><label>Email (optional)</label><input name="email" type="email"></div>
        <div class="field"><label>Valid for (hours)</label><input name="hours" type="number" value="48"></div>
        <button type="submit">Generate Link</button>
      </form>
    </div>
    <div class="card"><h3>Interviews</h3><table>{table}</table></div>
    """
    return page("Admin", body)


@app.route("/admin/download/<token>")
def admin_download(token):
    if request.args.get("pw") != config.ADMIN_PASSWORD:
        abort(403)
    row = db.get_interview(token)
    if not row or not row["result_path"]:
        abort(404)
    return send_file(row["result_path"], as_attachment=True)


# ---------------- Candidate interview ----------------
PROFILE_FIELDS = ["full_name", "age", "years_experience", "location", "notice_period", "expected_salary", "specialist"]


@app.route("/interview/<token>", methods=["GET", "POST"])
def interview(token):
    row = db.get_interview(token)
    if not row:
        return page("Interview", alert("Invalid interview link.", "error"))
    if row["status"] == "completed":
        return page("Interview", alert("This interview has already been completed.", "error"))
    if datetime.now() > datetime.fromisoformat(row["expires_at"]):
        return page("Interview", alert("This interview link has expired.", "error"))

    stage = request.form.get("stage")

    if request.method == "POST" and stage == "profile":
        profile = {k: request.form.get(k, "").strip() for k in PROFILE_FIELDS}
        questions = llm.generate_questions(profile)
        db.update_interview(token, status="in_progress", profile=json.dumps(profile), questions=json.dumps(questions))
        row = db.get_interview(token)

    elif request.method == "POST" and stage == "answers":
        questions = json.loads(row["questions"])
        answers = {q: request.form.get(f"a{i}", "").strip() for i, q in enumerate(questions)}
        profile = json.loads(row["profile"])
        evaluation = llm.evaluate(profile, answers)
        path = results.save_txt(row["name"], profile, answers, evaluation)
        fe = evaluation.get("final_evaluation", {})
        db.update_interview(token, status="completed", answers=json.dumps(answers), result_path=path,
                             score=fe.get("average_score"), recommendation=fe.get("recommendation"))
        notify_admin(row["name"])
        return page("Interview", alert("✅ Thank you! Your interview is complete.", "success"))

    if row["status"] == "pending":
        fields = "".join(
            f'<div class="field"><label>{f.replace("_", " ").title()}</label>'
            f'<input name="{f}" type="{"number" if f in ("age", "years_experience") else "text"}" required></div>'
            for f in PROFILE_FIELDS
        )
        body = f"""
        <div class="card">
          <h3>Welcome, {esc(row['name'])} 👋</h3>
          <p>Please fill your profile to begin. This link can only be used once.</p>
          <form method="post">
            <input type="hidden" name="stage" value="profile">
            {fields}
            <button type="submit">Start Interview</button>
          </form>
        </div>"""
    else:
        questions = json.loads(row["questions"])
        qfields = "".join(
            f'<div class="field"><label>Q{i + 1}. {esc(q)}</label>'
            f'<textarea name="a{i}" rows="4" required></textarea></div>'
            for i, q in enumerate(questions)
        )
        body = f"""
        <div class="card">
          <h3>Interview Questions</h3>
          <form method="post">
            <input type="hidden" name="stage" value="answers">
            {qfields}
            <button type="submit">Submit Answers</button>
          </form>
        </div>"""

    return page("Interview", body)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)

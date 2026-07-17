import html as _html

CSS = """
<style>
  :root { --accent:#4f46e5; --bg:#f5f6fa; --card:#fff; --text:#1f2430; --muted:#6b7280; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg);
         color:var(--text); margin:0; padding:40px 20px; }
  .wrap { max-width:700px; margin:0 auto; }
  h1 { font-size:1.6rem; margin-bottom:4px; }
  h3 { margin-top:0; }
  .card { background:var(--card); border-radius:14px; padding:28px;
          box-shadow:0 4px 20px rgba(0,0,0,.06); margin-bottom:24px; }
  label { display:block; font-weight:600; margin-bottom:6px; font-size:.9rem; }
  .field { margin-bottom:18px; }
  input, textarea { width:100%; padding:10px; border:1px solid #dfe1ea; border-radius:8px;
                     font-family:inherit; font-size:.95rem; }
  textarea { resize:vertical; }
  button { background:var(--accent); color:#fff; border:none; padding:12px 24px; border-radius:8px;
           font-size:1rem; font-weight:600; cursor:pointer; width:100%; }
  button:hover { background:#4338ca; }
  a.btn { color:var(--accent); font-weight:600; text-decoration:none; }
  table { width:100%; border-collapse:collapse; font-size:.85rem; }
  th, td { text-align:left; padding:8px; border-bottom:1px solid #eee; }
  .badge { padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:700; }
  .pending { background:#fef3c7; color:#92400e; }
  .in_progress { background:#dbeafe; color:#1d4ed8; }
  .completed { background:#dcfce7; color:#15803d; }
  .alert { padding:12px 16px; border-radius:8px; margin-bottom:16px; }
  .alert.error { background:#fee2e2; color:#b91c1c; }
  .alert.info { background:#e0e7ff; color:#3730a3; }
  .alert.success { background:#dcfce7; color:#15803d; }
  .score { font-weight:700; }
</style>
"""


def page(title, body):
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>{CSS}</head>
<body><div class="wrap"><h1>🩺 HR Interview Evaluator</h1>{body}</div></body></html>"""


def alert(msg, kind="info"):
    return f'<div class="alert {kind}">{msg}</div>'


def esc(s):
    return _html.escape(str(s or ""))

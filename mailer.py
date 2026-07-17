import smtplib
from email.mime.text import MIMEText
import config


def notify_admin(candidate_name):
    if not (config.EMAIL_USER and config.EMAIL_PASSWORD and config.ADMIN_EMAIL):
        return
    try:
        msg = MIMEText(f"Interview completed for {candidate_name}. Check the admin panel for results.")
        msg["Subject"] = f"Interview Completed - {candidate_name}"
        msg["From"] = config.EMAIL_USER
        msg["To"] = config.ADMIN_EMAIL
        with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(config.EMAIL_USER, config.EMAIL_PASSWORD)
            s.send_message(msg)
    except Exception as e:
        print("Email error:", e)

from datetime import datetime
from pathlib import Path
import config


def save_txt(name, profile, answers, evaluation):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = (name or "candidate").replace(" ", "_")
    path = Path(config.OUTPUT_DIR) / f"{safe}_{ts}.txt"

    lines = [f"Candidate: {name}", f"Date: {datetime.now():%Y-%m-%d %H:%M}", "=" * 50, "", "Profile:"]
    lines += [f"  {k}: {v}" for k, v in profile.items()]
    lines += ["", "Answers:"]
    for i, (q, a) in enumerate(answers.items(), 1):
        lines += [f"Q{i}: {q}", f"A{i}: {a}", ""]

    lines.append("Evaluation:")
    if "error" in evaluation:
        lines.append(f"  ERROR: {evaluation['error']}")
    else:
        for q, d in evaluation.get("evaluations", {}).items():
            lines.append(f"  {q} - Score {d.get('score')}: {d.get('strengths')} / {d.get('weaknesses')}")
        fe = evaluation.get("final_evaluation", {})
        lines += [
            "",
            f"Average Score: {fe.get('average_score')}",
            f"Recommendation: {fe.get('recommendation')}",
            f"Strengths: {fe.get('overall_strengths')}",
            f"Weaknesses: {fe.get('overall_weaknesses')}",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)

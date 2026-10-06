"""Daily digest delivery by Telegram and/or email. Both are free.

Telegram:  TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID
Email:     SMTP_USER and SMTP_PASSWORD (a Gmail app password works) and DIGEST_EMAIL.
           SMTP_HOST and SMTP_PORT default to Gmail.

Keep these in GitHub Actions secrets, not in profiles.json, which is public. A profile can
still set its own telegram_chat_id or email to send to someone else.
"""
from __future__ import annotations

import html
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd
import requests

from radar.match import Profile

log = logging.getLogger(__name__)
APP_URL = os.environ.get("APP_URL", "https://sa-tender-radar.streamlit.app")


def _date(ts) -> str:
    return "" if pd.isna(ts) else pd.Timestamp(ts).strftime("%a %d %b, %H:%M")


def telegram_text(profile: Profile, new: pd.DataFrame, closing: pd.DataFrame) -> str:
    e = html.escape
    lines = [f"<b>Tender Radar · {e(profile.name)}</b>",
             f"{len(new)} new match{'es' if len(new) != 1 else ''} today"]
    for r in new.itertuples():
        lines += ["", f"<b>{e(r.description[:160])}</b>",
                  f"{e(r.buyer)} · {e(r.province)}",
                  f"Score {r.score:.0f} · closes {_date(r.closes)}"]
        if r.briefing_compulsory and pd.notna(r.briefing_date):
            lines.append(f"Compulsory briefing {_date(r.briefing_date)}")
        lines.append(f'<a href="{APP_URL}/?tender={r.id}">Open</a> · ref {e(r.reference)}')
    if len(closing):
        lines += ["", "<b>Closing in the next 3 days</b>"]
        lines += [f"• {e(r.description[:90])} · {_date(r.closes)}" for r in closing.itertuples()]
    return "\n".join(lines)


def email_html(profile: Profile, new: pd.DataFrame, closing: pd.DataFrame) -> str:
    e = html.escape
    rows = "".join(
        f"""<tr><td style="padding:14px 0;border-bottom:1px solid #e5e7eb">
        <div style="font-size:15px;font-weight:600;color:#111827">{e(r.description)}</div>
        <div style="font-size:13px;color:#4b5563;margin-top:4px">{e(r.buyer)} · {e(r.province)}
        · ref {e(r.reference)}</div>
        <div style="font-size:13px;color:#4b5563;margin-top:2px">Closes {_date(r.closes)}
        · fit score {r.score:.0f}/100</div>
        <a href="{APP_URL}/?tender={r.id}" style="font-size:13px;color:#0f5c4a">View tender</a>
        </td></tr>""" for r in new.itertuples())
    soon = "".join(f"<li>{e(r.description[:120])} — {_date(r.closes)}</li>"
                   for r in closing.itertuples())
    return f"""<div style="font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:640px">
    <div style="font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#6b7280">
    Tender Radar</div>
    <h2 style="margin:4px 0 16px;color:#111827">{len(new)} new tenders for {e(profile.name)}</h2>
    <table style="width:100%;border-collapse:collapse">{rows}</table>
    {f'<h3 style="color:#111827">Closing in the next 3 days</h3><ul>{soon}</ul>' if soon else ''}
    <p style="font-size:12px;color:#6b7280">Source: eTenders, National Treasury.
    Always check the official tender document before bidding.</p></div>"""


def send_telegram(chat_id: str, text: str) -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not (token and chat_id):
        return
    # Telegram caps messages at 4096 characters; split on blank lines.
    chunks, current = [], ""
    for block in text.split("\n\n"):
        if len(current) + len(block) + 2 > 4000:
            chunks.append(current)
            current = ""
        current += ("\n\n" if current else "") + block
    chunks.append(current)
    for chunk in chunks:
        resp = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                             json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML",
                                   "disable_web_page_preview": True}, timeout=30)
        if not resp.ok:
            log.warning("Telegram error: %s", resp.text[:200])


def send_email(to: str, subject: str, body_html: str) -> None:
    user, password = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASSWORD")
    if not (user and password and to):
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.attach(MIMEText(body_html, "html", "utf-8"))
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
        smtp.login(user, password)
        smtp.sendmail(user, [to], msg.as_string())


def deliver(profile: Profile, new: pd.DataFrame, closing: pd.DataFrame) -> list[str]:
    sent = []
    if new.empty and closing.empty:
        return sent
    chat_id = profile.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    email = profile.email or os.environ.get("DIGEST_EMAIL", "")
    if chat_id and os.environ.get("TELEGRAM_BOT_TOKEN"):
        send_telegram(chat_id, telegram_text(profile, new, closing))
        sent.append("telegram")
    if email and os.environ.get("SMTP_USER"):
        send_email(email, f"{len(new)} new tenders for {profile.name}",
                   email_html(profile, new, closing))
        sent.append("email")
    return sent

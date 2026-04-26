import asyncio
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from smtplib import SMTPAuthenticationError
from typing import Any, Dict

import PyPDF2
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai.types import Content, Part
from pydantic import EmailStr, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: SecretStr = Field(...)
    sender_email: EmailStr = Field(...)
    email_app_password: SecretStr = Field(...)
    resume_path: str = Field("my_resume.pdf")
    notice_period_days: int = Field(90)
    min_experience_years: int = Field(0)
    max_experience_years: int = Field(2)
    max_jobs_to_send: int = Field(5)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = Settings()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def normalize_gmail_app_password(password: str) -> str:
    return "".join(password.strip().strip("'\"").split())


def validate_email_settings() -> bool:
    normalized_password = normalize_gmail_app_password(
        config.email_app_password.get_secret_value()
    )
    if len(normalized_password) != 16:
        logging.error(
            "EMAIL_APP_PASSWORD must be a 16-character Gmail App Password after removing spaces. "
            "Current normalized length: %s. Generate a new one from Google Account -> Security -> "
            "2-Step Verification -> App Passwords.",
            len(normalized_password),
        )
        return False
    return True


def extract_resume_data(file_path: str = config.resume_path) -> str:
    logging.info("Reading resume...")
    if not os.path.exists(file_path):
        return f"Error: Resume not found at '{file_path}'."

    text = ""
    try:
        with open(file_path, "rb") as file:
            for page in PyPDF2.PdfReader(file).pages:
                text += page.extract_text() or ""
        return f"Resume Content: {text[:7000]}"
    except Exception as e:
        return f"System Error: {str(e)}"


def dispatch_email_report(html_job_report: str) -> Dict[str, Any]:
    logging.info("Formatting and sending email...")
    sender = config.sender_email
    sender_id = str(config.sender_email)
    password = normalize_gmail_app_password(config.email_app_password.get_secret_value())

    if "```html" in html_job_report:
        html_job_report = html_job_report.split("```html")[1].split("```")[0].strip()

    msg = MIMEMultipart()
    msg["From"] = msg["To"] = sender
    msg["Subject"] = f"Daily Product-Based Job Matches ({config.notice_period_days}-Day Notice)"
    msg.attach(MIMEText(html_job_report, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_id, password)
            server.send_message(msg)
        logging.info("Email successfully delivered.")
        return {"status": "success", "message": "Email delivered."}
    except SMTPAuthenticationError as e:
        logging.error(
            "Email login failed for '%s'. Gmail requires the account email plus a valid App Password "
            "(not the normal Gmail password). Original error: %s",
            sender_id,
            e,
        )
        return {
            "status": "error",
            "message": "Gmail authentication failed. Check SENDER_EMAIL/SENDER_ID and EMAIL_APP_PASSWORD.",
        }
    except Exception as e:
        logging.error("Email failed: %s", e)
        return {"status": "error", "message": str(e)}


def build_agent() -> Agent:
    os.environ["GEMINI_API_KEY"] = config.gemini_api_key.get_secret_value()
    return Agent(
        name="Auto_Job_Hunter",
        model="gemini-2.5-flash",
        instruction=f"""
        You are an elite AI recruiter helping the user transition to start-ups or product-based companies.

        Your goal is to find only accurate, relevant, genuine software roles for this specific candidate.

        Follow this process:
        1. Read the resume carefully and infer the candidate profile:
           - core languages, frameworks, databases, cloud/tools, internships/projects
           - likely target roles based on the actual resume
           - seniority fit for an early-career candidate
        2. Search only for recent openings that fit this candidate and are suitable for
           {config.min_experience_years}-{config.max_experience_years} years of experience.
        3. Prioritize start-ups ,product-based companies and genuine roles from:
           - official company careers pages
           - well-known trusted job boards when the listing is clearly tied to a real company
        4. Verify each role before including it:
           - the role must still appear open/live
           - the company must be identifiable
           - the apply URL must be direct and usable
           - the role should clearly fit an early-career profile
           - reject vague, duplicate, suspicious, staffing-only, or clearly outdated posts
        5. Prefer roles that align strongly with the resume's actual skills and projects.
        6. Keep search queries specific. Include terms like:
           - software engineer OR backend engineer OR full stack engineer OR associate software engineer
           - {config.min_experience_years}-{config.max_experience_years} years
           - fresher OR entry level OR early career when useful
           - "{config.notice_period_days} days notice period" OR "3 months notice period" when useful

        Output requirements:
        - Return ONLY raw HTML. No markdown.
        - Include up to {config.max_jobs_to_send} roles, sorted by best match first.
        - If fewer than {config.max_jobs_to_send} trustworthy matches are found, return fewer roles instead of guessing.
        - For each role include:
          Company, Title, Location, Experience Range, Why It Matches The Resume, Verification Note, Apply URL
        - Add a short candidate summary at the top based on the resume.
        - In the verification note, briefly state why you believe the role is genuine
          such as official careers page, recognizable company listing, or live direct application page.
        """,
        tools=[google_search],
    )


async def execute_daily_hunt():
    if not validate_email_settings():
        return

    resume_text = extract_resume_data()
    if "Error" in resume_text:
        logging.error(resume_text)
        return

    agent = build_agent()
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=agent.name, user_id="admin")
    runner = Runner(agent=agent, session_service=session_service, app_name=agent.name)

    prompt = (
        f"""Here is my resume data:

        {resume_text}

        Run the daily job hunt with these constraints:
        - Target only roles relevant to my actual resume and skills.
        - Focus on {config.min_experience_years}-{config.max_experience_years} years of experience.
        - Prioritize software engineer, backend engineer, full stack engineer, and related entry-level start-up or product-company roles.
        - My notice period is {config.notice_period_days} days.
        - Prefer genuine, live openings that you can verify from trustworthy sources.
        - Do not include doubtful or weakly related jobs just to fill the list.
        """
    )

    final_html_response = ""
    logging.info("Agent is searching the web for matching jobs...")

    try:
        async for event in runner.run_async(
            user_id="admin",
            session_id=session.id,
            new_message=Content(parts=[Part.from_text(text=prompt)], role="user"),
        ):
            if event.is_final_response():
                final_html_response = event.content.parts[0].text
    except Exception as e:
        logging.error("Agent Search Failure: %s", e)
        return

    if final_html_response:
        dispatch_email_report(final_html_response)
    else:
        logging.warning("Agent returned an empty response. No email sent.")


if __name__ == "__main__":
    asyncio.run(execute_daily_hunt())

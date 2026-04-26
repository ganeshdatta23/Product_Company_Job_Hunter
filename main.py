import asyncio
import json
import logging
import os
import re
import smtplib
from urllib.parse import urlparse
from typing import List
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import PyPDF2
import requests
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai.types import Content, Part
from pydantic import BaseModel, EmailStr, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: SecretStr = Field(..., validation_alias="GEMINI_API_KEY")
    sender_email: EmailStr = Field(..., validation_alias="SENDER_EMAIL")
    email_app_password: SecretStr = Field(..., validation_alias="EMAIL_APP_PASSWORD")

    resume_path: str = "my_resume.pdf"
    max_jobs: int = 5
    min_confidence: int = 70

config = Settings()

logging.basicConfig(level=logging.INFO)


class Job(BaseModel):
    company: str
    title: str
    location: str
    apply_url: str
    why_match: str
    confidence: int

class JobReport(BaseModel):
    candidate_summary: str
    jobs: List[Job]


def extract_resume():
    text = ""
    with open(config.resume_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text[:8000]


def is_direct_job_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return False, "invalid-url-format"

    query = parsed.query.lower()
    path = parsed.path.lower()
    netloc = parsed.netloc.lower()
    url_lower = url.lower()

    if any(token in query for token in ("q=", "query=", "keyword=", "keywords=", "search=")):
        return False, "search-results-url"

    if any(marker in url_lower for marker in ("/jobs?", "/search", "/results", "/find-jobs")):
        return False, "listing-page-url"

    if "indeed." in netloc and "/viewjob" not in path:
        return False, "indeed-non-direct-url"

    return True, "direct-url-shape"


def is_trusted_job_host(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    trusted_patterns = (
        ("wellfound.com", "/jobs/"),
        ("boards.greenhouse.io", "/"),
        ("job-boards.greenhouse.io", "/"),
        ("lever.co", "/"),
        ("jobs.lever.co", "/"),
        ("myworkdayjobs.com", "/"),
        ("workdayjobs.com", "/"),
        ("smartrecruiters.com", "/job/"),
        ("ashbyhq.com", "/job/"),
    )

    return any(domain in host and path_prefix in path for domain, path_prefix in trusted_patterns)


def validate_url(url: str) -> tuple[bool, str]:
    is_direct, direct_reason = is_direct_job_url(url)
    if not is_direct:
        return False, direct_reason

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/135.0.0.0 Safari/537.36"
        )
    }

    try:
        res = requests.head(url, timeout=10, allow_redirects=True, headers=headers)
        if res.status_code < 400:
            return True, f"head-{res.status_code}"
        if res.status_code in {403, 429} and is_trusted_job_host(url):
            return True, f"trusted-host-head-{res.status_code}"
        if res.status_code not in {403, 405, 406, 429}:
            return False, f"head-{res.status_code}"
    except requests.RequestException as exc:
        logging.info("HEAD check failed for %s: %s", url, exc)

    try:
        res = requests.get(url, timeout=10, allow_redirects=True, headers=headers, stream=True)
        if res.status_code < 400:
            return True, f"get-{res.status_code}"
        if res.status_code in {403, 429} and is_trusted_job_host(url):
            return True, f"trusted-host-get-{res.status_code}"
        return False, f"get-{res.status_code}"
    except requests.RequestException as exc:
        return False, f"request-error-{exc.__class__.__name__}"


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

def build_agent():
    os.environ["GEMINI_API_KEY"] = config.gemini_api_key.get_secret_value()

    return Agent(
        name="JobHunter",
        model="gemini-2.5-flash",
        instruction=f"""
You are a STRICT job matching system.

CRITICAL RULES:
- ONLY return jobs found via search tool
- NEVER guess or invent jobs
- EVERY job MUST have a direct apply URL to the specific job post
- If unsure, do not include
- Prefer fewer accurate jobs over many
- NEVER return search results pages, listing pages, or generic job board query URLs
- DO NOT return URLs like indeed.com/jobs?... , company homepages, or careers homepages
- If you cannot find the exact job-post URL, omit the job

SOURCE PRIORITY:
1. Official company careers
2. ATS (Greenhouse, Lever, Workday)
3. Trusted job boards ONLY if legit

FILTER:
- Entry level (0-2 years)
- Software / Backend / Full Stack roles only
- Reject outdated or vague jobs

OUTPUT FORMAT (JSON ONLY):
{{
  "candidate_summary": "...",
  "jobs": [
    {{
      "company": "...",
      "title": "...",
      "location": "...",
      "apply_url": "...",
      "why_match": "...",
      "confidence": 0-100
    }}
  ]
}}

RETURN MAX {config.max_jobs} JOBS
CONFIDENCE MUST BE >= 70
"""
        ,
        tools=[google_search]
    )

def send_email(html):
    msg = MIMEMultipart()
    msg["From"] = str(config.sender_email)
    msg["To"] = str(config.sender_email)
    msg["Subject"] = "Daily Job Matches"

    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(
            str(config.sender_email),
            config.email_app_password.get_secret_value()
        )
        server.send_message(msg)


def render_html(report: JobReport):
    html = f"<h2>Job Matches</h2><p>{report.candidate_summary}</p>"

    for job in report.jobs:
        html += f"""
        <div style='border:1px solid #ccc;padding:10px;margin:10px'>
            <h3>{job.title}</h3>
            <p><b>{job.company}</b> - {job.location}</p>
            <p>{job.why_match}</p>
            <p>Confidence: {job.confidence}</p>
            <a href="{job.apply_url}">Apply</a>
        </div>
        """

    return html

async def main():
    resume = extract_resume()

    agent = build_agent()
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="JobHunter", user_id="u1")

    runner = Runner(agent=agent, session_service=session_service, app_name="JobHunter")

    prompt = f"""
Analyze this resume and find matching jobs:

{resume}

STRICT:
- Only real jobs
- Use search tool
- No guessing
"""

    response_text = ""

    async for event in runner.run_async(
        user_id="u1",
        session_id=session.id,
        new_message=Content(parts=[Part.from_text(text=prompt)], role="user"),
    ):
        if event.is_final_response():
            response_text = event.content.parts[0].text

    try:
        data = json.loads(extract_json(response_text))
        report = JobReport(**data)
    except Exception as e:
        logging.error("Parsing failed: %s", e)
        return

    logging.info("Model returned %s jobs before filtering", len(report.jobs))

    filtered_jobs = []
    for job in report.jobs:
        if job.confidence < config.min_confidence:
            logging.info(
                "Rejected job '%s' at %s: confidence %s < %s",
                job.title,
                job.company,
                job.confidence,
                config.min_confidence,
            )
            continue

        url_ok, url_reason = validate_url(job.apply_url)
        if not url_ok:
            logging.info(
                "Rejected job '%s' at %s: URL validation failed (%s) for %s",
                job.title,
                job.company,
                url_reason,
                job.apply_url,
            )
            continue

        logging.info(
            "Accepted job '%s' at %s: confidence=%s, url_check=%s",
            job.title,
            job.company,
            job.confidence,
            url_reason,
        )
        filtered_jobs.append(job)

    report.jobs = filtered_jobs[:config.max_jobs]
    logging.info("Accepted %s jobs after filtering", len(report.jobs))

    if not report.jobs:
        logging.info("Raw model response: %s", response_text)
        logging.warning("No valid jobs found")
        return

    html = render_html(report)
    logging.info("Sending email with %s jobs", len(report.jobs))
    send_email(html)


if __name__ == "__main__":
    asyncio.run(main())

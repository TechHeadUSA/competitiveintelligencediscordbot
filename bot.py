# bot.py
import os
import time
import asyncio
import logging
import re
from typing import Dict, List, Dict as _Dict

from dotenv import load_dotenv
from openai import OpenAI
import discord
from discord import app_commands

# Optional: external calls for news and site content
import aiohttp

# ---------- Setup ----------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

if not all([OPENAI_KEY, DISCORD_TOKEN, ASSISTANT_ID]):
    raise SystemExit("Missing required env vars: OPENAI_KEY, DISCORD_TOKEN, ASSISTANT_ID")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
openai_client = OpenAI(api_key=OPENAI_KEY)

# Keep one assistant thread per Discord channel (or per user, if you prefer)
CHANNEL_THREAD_CACHE: Dict[int, str] = {}

# ---------- Competitors & domains ----------
DEFAULT_COMPETITORS: Dict[str, List[str]] = {
    "vmware": ["vmware.com", "broadcom.com"],
    "nutanix": ["nutanix.com"],
    "aws": ["aws.amazon.com", "aws.amazon.com/eks", "aws.amazon.com/ec2"],
    "azure": ["azure.microsoft.com", "learn.microsoft.com/azure/openshift"],
    "google": ["cloud.google.com", "cloud.google.com/kubernetes-engine"],
    "oracle": ["oracle.com/cloud"],
    "suse": ["suse.com", "rancher.com"],
    # include Red Hat for parity/comparisons
    "redhat": ["redhat.com/openshift/virtualization", "redhat.com/openshift"],
}

def infer_competitors(question: str) -> List[str]:
    q = (question or "").lower()
    hits = [k for k in DEFAULT_COMPETITORS.keys() if k in q]
    return hits or list(DEFAULT_COMPETITORS.keys())

# ---------- Lightweight research layer ----------
async def bing_news_search(session: aiohttp.ClientSession, query: str, count: int = 8) -> List[_Dict]:
    """Return [{title, url, source}] via Bing News API v7. If no key, return []."""
    if not BING_KEY:
        return []
    params = {"q": query, "count": count, "freshness": "Week", "mkt": "en-US", "textDecorations": False}
    headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
    try:
        async with session.get("https://api.bing.microsoft.com/v7.0/news/search",
                               params=params, headers=headers, timeout=20) as r:
            if r.status != 200:
                logging.warning(f"Bing News status {r.status} for query: {query}")
                return []
            data = await r.json()
            out = []
            for it in data.get("value", []):
                out.append({
                    "title": it.get("name", ""),
                    "url": it.get("url", ""),
                    "source": (it.get("provider") or [{}])[0].get("name", ""),
                })
            return out
    except Exception as e:
        logging.warning(f"Bing News error for '{query}': {e}")
        return []

def _simple_html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", html)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

async def fetch_url_text(session: aiohttp.ClientSession, url: str) -> str:
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                return ""
            html = await r.text(errors="ignore")
            return _simple_html_to_text(html)[:20000]
    except Exception:
        return ""

async def gather_research(question: str) -> List[_Dict]:
    """
    Return a list of dicts: {title, url, source, text}
    Pulls:
      - Bing News (if key present) for each inferred competitor + the question
      - Official competitor site homepages for latest messaging
    """
    comp = infer_competitors(question)
    queries = [f"{c} virtualization Kubernetes OpenShift competitor news" for c in comp]
    queries.append(f"Red Hat OpenShift Virtualization {question or ''}".strip())

    results: List[_Dict] = []
    async with aiohttp.ClientSession() as session:
        # News
        for q in queries:
            hits = await bing_news_search(session, q)
            results.extend(hits)

        # Official sites
        for c in comp:
            for domain in DEFAULT_COMPETITORS.get(c, []):
                url = f"https://{domain}" if not domain.startswith("http") else domain
                results.append({"title": f"{c} site: {domain}", "url": url, "source": domain})

        # Dedupe by URL (preserve order)
        seen = set()
        deduped = []
        for it in results:
            u = it.get("url", "")
            if u and u not in seen:
                seen.add(u)
                deduped.append(it)

        # Fetch bodies (cap to reduce latency)
        final: List[_Dict] = []
        for it in deduped[:15]:
            body = await fetch_url_text(session, it["url"])
            if body:
                it["text"] = body
                final.append(it)
        return final

# ---------- Assistants helpers ----------
def build_run_instructions() -> str:
    return (
        "You are a Competitive Intelligence assistant for Red Hat OpenShift Virtualization.\n"
        "Use ONLY the provided 'Research Documents' to answer. If details are missing, say so.\n"
        "Output format:\n"
        "1) Executive Summary (3–5 bullets)\n"
        "2) Key Insights (per competitor, concise bullets)\n"
        "3) Risks & Opportunities\n"
        "4) Sources (list raw URLs used)\n"
        "Rules: Be factual and vendor-neutral. No speculation without evidence. Always include sources."
    )

def format_docs_for_model(docs: List[_Dict]) -> str:
    parts = []
    for d in docs:
        parts.append(
            f"TITLE: {d.get('title','')}\n"
            f"URL: {d.get('url','')}\n"
            f"EXCERPT:\n{(d.get('text','') or '')[:3000]}\n---"
        )
    return "\n".join(parts)

async def get_or_create_thread(channel_id: int) -> str:
    if channel_id in CHANNEL_THREAD_CACHE:
        return CHANNEL_THREAD_CACHE[channel_id]
    th = openai_client.beta.threads.create()
    CHANNEL_THREAD_CACHE[channel_id] = th.id
    return th.id

def run_assistant_blocking(thread_id: str, user_prompt: str, docs: List[_Dict]) -> str:
    """Blocking portion (run in a thread): create messages, run, return the latest assistant text."""
    # Add user message with research bundle
    bundle = format_docs_for_model(docs)
    prompt = f"Question:\n{user_prompt}\n\nResearch Documents:\n{bundle}"

    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )

    run = openai_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        instructions=build_run_instructions()  # ✅ correct place for per-run “system” guidance
    )

    start = time.time()
    # Poll for completion (simple, robust)
    while True:
        status = openai_client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if status.status == "completed":
            break
        if status.status in {"failed", "cancelled", "expired"}:
            raise RuntimeError(f"Assistant run ended with status: {status.status}")
        if time.time() - start > 70:
            raise TimeoutError("Assistant run timed out")
        time.sleep(1.0)

    msgs = openai_client.beta.threads.messages.list(thread_id=thread_id)
    for m in msgs.data:
        if m.role == "assistant":
            if m.content and m.content[0].type == "text":
                return m.content[0].text.value
    return "I couldn't produce a response this time."

# ---------- Discord bot ----------
intents = discord.Intents.default()
# message_content not required for slash commands, but safe if you later add prefix commands:
intents.message_content = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

@bot.event
async def on_ready():
    await tree.sync()
    logging.info(f"Logged in as {bot.user}")

@tree.command(name="ask", description="Ask about OpenShift Virtualization and competitors")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)  # visible 'thinking…'

    try:
        docs = await gather_research(question)
        if not docs:
            await interaction.followup.send(
                "I couldn’t gather sources right now. Please try again in a minute or rephrase your question."
            )
            return

        thread_id = await get_or_create_thread(interaction.channel_id)
        answer = await asyncio.to_thread(run_assistant_blocking, thread_id, question, docs)

        # Discord hard cap ~2000 chars; keep margin
        if len(answer) > 1900:
            answer = answer[:1900] + "\n… (truncated)"

        await interaction.followup.send(answer)

    except Exception as e:
        logging.exception("Error handling /ask: %s", e)
        await interaction.followup.send("Something went wrong while researching or generating the answer.")

if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)

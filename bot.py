import os
import time
import asyncio
import logging
from typing import Dict

from dotenv import load_dotenv
from openai import OpenAI
import discord
from discord import app_commands

# ---------- Setup ----------
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

if not all([OPENAI_KEY, DISCORD_TOKEN, ASSISTANT_ID]):
    raise SystemExit("Missing required env vars: OPENAI_KEY, DISCORD_TOKEN, ASSISTANT_ID")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
openai_client = OpenAI(api_key=OPENAI_KEY)

# Keep one assistant thread per Discord channel
CHANNEL_THREAD_CACHE: Dict[int, str] = {}

# ---------- Assistant Configuration ----------
def get_system_instructions() -> str:
    """
    Core identity and operating instructions for the assistant.
    Configure this in the OpenAI Assistant dashboard OR pass it per-run.
    """
    return """You are a Product Marketing Manager for Red Hat OpenShift Virtualization.

YOUR ROLE:
- Research and provide validated, high-quality competitive intelligence
- Focus on OpenShift Virtualization vs. competitors: VMware, Nutanix, AWS, Azure, Google Cloud, Oracle, SUSE/Rancher
- Be factual, objective, and evidence-based
- Always cite your sources with URLs

OUTPUT FORMAT:
1. **Executive Summary** (3-5 key bullets)
2. **Competitive Insights** (organized by competitor when relevant)
3. **Risks & Opportunities** (strategic implications)
4. **Sources** (list all URLs referenced)

GUIDELINES:
- Use web search to find current information when needed
- Be vendor-neutral in analysis - let facts speak
- If information is unavailable or uncertain, say so explicitly
- Focus on technical capabilities, pricing, market positioning, and recent news
- Highlight Red Hat's strengths without being promotional
- No speculation - only cite verifiable information"""

# ---------- Assistants API Helpers ----------
async def get_or_create_thread(channel_id: int) -> str:
    """Get existing thread for channel or create new one."""
    if channel_id in CHANNEL_THREAD_CACHE:
        return CHANNEL_THREAD_CACHE[channel_id]
    
    thread = openai_client.beta.threads.create()
    CHANNEL_THREAD_CACHE[channel_id] = thread.id
    logging.info(f"Created new thread {thread.id} for channel {channel_id}")
    return thread.id

def run_assistant_blocking(thread_id: str, user_question: str) -> str:
    """
    Run the assistant (blocking call - use in thread pool).
    Returns the assistant's response text.
    """
    # Add user message
    openai_client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_question
    )
    
    # Create and run
    run = openai_client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        instructions=get_system_instructions()
    )
    
    # Poll for completion
    start = time.time()
    while True:
        status = openai_client.beta.threads.runs.retrieve(
            thread_id=thread_id, 
            run_id=run.id
        )
        
        if status.status == "completed":
            break
        
        if status.status in {"failed", "cancelled", "expired"}:
            error_msg = getattr(status, 'last_error', None)
            raise RuntimeError(f"Assistant run {status.status}: {error_msg}")
        
        if time.time() - start > 120:  # 2 minute timeout
            openai_client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            raise TimeoutError("Assistant took too long to respond")
        
        time.sleep(1.5)
    
    # Get response
    messages = openai_client.beta.threads.messages.list(thread_id=thread_id, limit=1)
    for msg in messages.data:
        if msg.role == "assistant":
            for content in msg.content:
                if content.type == "text":
                    return content.text.value
    
    return "I couldn't generate a response. Please try rephrasing your question."

# ---------- Discord Bot ----------
intents = discord.Intents.default()
intents.message_content = True

bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

@bot.event
async def on_ready():
    await tree.sync()
    logging.info(f"✅ Logged in as {bot.user}")
    logging.info(f"📊 Assistant ID: {ASSISTANT_ID}")

@tree.command(
    name="ask",
    description="Ask competitive intelligence questions about OpenShift Virtualization"
)
async def ask(interaction: discord.Interaction, question: str):
    """Main command - ask the CI assistant a question."""
    await interaction.response.defer(thinking=True)
    
    try:
        # Get or create thread for this channel
        thread_id = await get_or_create_thread(interaction.channel_id)
        
        # Run assistant (in thread pool to avoid blocking)
        answer = await asyncio.to_thread(
            run_assistant_blocking, 
            thread_id, 
            question
        )
        
        # Split long responses (Discord limit ~2000 chars)
        if len(answer) <= 1900:
            await interaction.followup.send(answer)
        else:
            # Split into chunks
            chunks = [answer[i:i+1900] for i in range(0, len(answer), 1900)]
            await interaction.followup.send(chunks[0])
            for chunk in chunks[1:]:
                await interaction.channel.send(chunk)
    
    except TimeoutError:
        await interaction.followup.send(
            "⏱️ The request took too long. Please try a more specific question."
        )
    except Exception as e:
        logging.exception("Error in /ask command")
        await interaction.followup.send(
            "❌ Something went wrong. Please try again or rephrase your question."
        )

@tree.command(
    name="reset",
    description="Clear conversation history for this channel"
)
async def reset(interaction: discord.Interaction):
    """Reset the conversation thread for this channel."""
    channel_id = interaction.channel_id
    
    if channel_id in CHANNEL_THREAD_CACHE:
        del CHANNEL_THREAD_CACHE[channel_id]
        await interaction.response.send_message(
            "✅ Conversation history cleared for this channel.",
            ephemeral=True
        )
    else:
        await interaction.response.send_message(
            "ℹ️ No conversation history to clear.",
            ephemeral=True
        )

@tree.command(
    name="help",
    description="Learn how to use the CI Assistant"
)
async def help_command(interaction: discord.Interaction):
    """Show help information."""
    help_text = """**Red Hat OpenShift Virtualization - Competitive Intelligence Assistant**

I'm your Product Marketing Manager specialized in competitive intelligence.

**Commands:**
• `/ask [question]` - Ask about competitors, features, pricing, market trends
• `/reset` - Clear conversation history for this channel
• `/help` - Show this message

**Example Questions:**
• "Compare OpenShift Virtualization to VMware vSphere"
• "What are Nutanix's latest announcements?"
• "How does our pricing compare to AWS?"
• "What are the key differentiators vs Azure?"

**Tips:**
• I maintain conversation context within each channel
• I search the web for current information
• All answers include source citations
• Be specific for better results"""
    
    await interaction.response.send_message(help_text, ephemeral=True)

if __name__ == "__main__":
    logging.info("🚀 Starting Red Hat CI Assistant Bot...")
    bot.run(DISCORD_TOKEN)

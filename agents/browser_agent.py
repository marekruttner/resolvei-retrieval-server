# /agents/browser_agent.py

import os
import asyncio
from dotenv import load_dotenv

# If you're using the Ollama wrapper from langchain_ollama:
# from langchain_ollama import ChatOllama
# If you're using the local LLM from your code, just import that instead

from browser_use.agent.service import Agent as BrowserAgent
from browser_use.browser.browser import Browser, BrowserConfig, BrowserContextConfig

# Just in case you need environment variables:
load_dotenv()

###############################################################################
# BROWSER AGENT CREATION
###############################################################################

def create_browser() -> Browser:
    """
    Create and configure the browser for multi-step tasks.
    Adjust headless, security, or viewport as needed.
    """
    return Browser(
        config=BrowserConfig(
            headless=True,  # set to False if you want to see the browser
            disable_security=False,
            new_context_config=BrowserContextConfig(
                viewport_expansion=-1,
                highlight_elements=False,
                # adjust times or window size as you see fit
                # browser_window_size={
                #     'width': 1280,
                #     'height': 1100,
                # },
            ),
        )
    )

async def run_browser_task(task: str, llm, max_steps: int = 20) -> str:
    """
    Launches a multi-step Agent with a user-specified task.
    The agent can click, type, navigate, etc., to accomplish the goal.
    This is an async function using 'browser-use'.
    """
    browser = create_browser()

    # Make sure your LLM is something that can produce chain-of-thought.
    # For example, a local LLM from 'langchain_ollama' or your own.
    agent = BrowserAgent(
        task=task,
        llm=llm,
        browser=browser,
        use_vision=False,  # set True if you want image-based tasks
        validate_output=False,
    )
    try:
        print(f"[BrowserAgent] Starting agent run for task: {task}")
        history = await agent.run(max_steps=max_steps)
        # 'history' is an AgentHistoryList with step-by-step actions
        return str(history)  # or history[-1] to get last step
    except Exception as e:
        print(f"[BrowserAgent Error]: {e}")
        return "Unable to complete the browser task."

def run_browser_task_sync(task: str, llm, max_steps: int = 20) -> str:
    """
    Synchronous wrapper around the async function for code that can't await.
    """
    return asyncio.run(run_browser_task(task, llm, max_steps))

###############################################################################
# JUDGER: DECIDE IF WE NEED BROWSER
###############################################################################

def should_use_browser(query: str, llm, llm_lock) -> bool:
    """
    Uses your local LLM to decide whether the query needs up-to-date or real-time info.
    Return True if we should do multi-step browsing, False otherwise.
    """
    judger_prompt = f"""
        System: You are a domain expert that decides if a multi-step web browsing agent is needed.
        If the user query references for example to specific device settings, realtime informations, answer "YES".
        Here are the examples of the "YES" queries:
            "Jak se připojím k Eduroam na zařízení s iOS?"
            "jak se připojím k wifi na telefonu samsung?"
            "proč mi nefunguje internet"
        
        Otherwise, answer "NO".

        User Query: "{query}"
        Answer with ONLY YES or NO.
        """.strip()

    with llm_lock:
        result = llm.invoke(judger_prompt).strip().upper()

    return (result == "YES")

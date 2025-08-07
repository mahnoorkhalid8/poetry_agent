import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.run import RunConfig
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent
import asyncio
import chainlit as cl

load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not presemt in .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model_provider=external_client,
    model=model,
    tracing_disabled=True
)

lyricalAgent = Agent(
    name="Lyrical Agent",
    instructions="You are an expert in Lyric Poetry. Analyze the poem deeply and give a poetic Tashreeh. Highlight emotions and feelings.",
    model=model
)

narrativeAgent = Agent(
    name="Narrative Agent",
    instructions="You are an expert in Narrative Poetry. Analyze the story, characters, and events in detail. Provide a thoughtful Tashreeh.",
    model=model
)

dramaticAgent = Agent(
    name="Dramatic Agent",
    instructions="You are an expert in Dramatic Poetry. Analyze it as if it is a performance piece. Explain the emotions, dialogues, and theatrical tone.",
    model=model
)

triage_agent = Agent(
    name="Poetry Triage Agent",
    instructions=(
        "You are a Poetry Triage Agent. Your job is to read the given poem (which can be in English or Urdu) and decide if it's:"
        "\n- Lyric Poetry (emotions, personal feelings),"
        "\n- Narrative Poetry (story, characters, events),"
        "\n- Dramatic Poetry (performance, dialogues, theatre style)."
        "\nAfter deciding, hand off the analysis to the correct agent: lyricalAgent, narrativeAgent, or dramaticAgent."
        "\nReturn only the final analysis in the same language as the poem."
    ),
    model=model,
    handoffs=[lyricalAgent, narrativeAgent, dramaticAgent]
)

@cl.on_message
async def message(message: cl.Message):
    poem = message.content
    await cl.Message(content="Analyzing your poem...").send()
    
    result = await Runner.run(
        starting_agent=triage_agent,
        input=poem,
        run_config=config
    )
    await cl.Message(content=f"Final Analysis:\n{result.final_output}").send()

  

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# Set up the paths
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ace_agent.sub_agents.generator import generator
from google.adk.runners import InMemoryRunner

from google.genai import types as genai_types

async def main():
    runner = InMemoryRunner(agent=generator)
    runner.auto_create_session = True
    print("Running Generator...")
    new_msg = genai_types.Content(role='user', parts=[genai_types.Part(text='What is 2+2?')])
    async for event in runner.run_async(user_id='test', session_id='test', new_message=new_msg):
        print(f"\n--- Event from: {event.author} ---")
        if event.content:
            for part in event.content.parts:
                if part.text: print(f"Text: {part.text}")
                if part.function_call: print(f"Tool Call: {part.function_call.name}({part.function_call.args})")
                if part.function_response: print(f"Tool Response: {part.function_response.name}: {part.function_response.response}")
            
    session = runner._get_session('test', 'test')
    print("\nState Keys:", session.state.keys())
    print("\nGenerator Output:", session.state.get('generator_output'))

if __name__ == "__main__":
    asyncio.run(main())

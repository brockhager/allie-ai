import asyncio
import sys
from pathlib import Path
# Ensure project root is on sys.path so 'backend' package can be imported when executing the script directly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from backend import server

async def run():
    p1 = {'prompt':'how big are the Rocky Mountains','max_tokens':60,'conversation_context':[]}
    r1 = await server.generate_response(p1)
    print('First response:')
    print(r1.get('text',''))

    ctx = [
        {'role':'me','text':'how big are the Rocky Mountains'},
        {'role':'them','text': r1.get('text','')}
    ]

    p2 = {'prompt':'how tall are they','max_tokens':60,'conversation_context':ctx}
    r2 = await server.generate_response(p2)
    print('\nFollow-up response:')
    print(r2.get('text',''))

if __name__=='__main__':
    asyncio.run(run())

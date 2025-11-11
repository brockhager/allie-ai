import json
from pathlib import Path

# Check conversations file
conv_file = Path('data/conversations.json')
if conv_file.exists():
    with open(conv_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    print(f'Total conversations: {len(conversations)}')

    # Check recent conversations
    recent_convs = conversations[-3:] if conversations else []
    for i, conv in enumerate(recent_convs):
        messages = conv.get('messages', [])
        print(f'\nConversation {len(conversations) - 2 + i}: {len(messages)} messages')
        for msg in messages[-2:]:  # Last 2 messages
            role = msg.get('role', 'unknown')
            text = msg.get('text', '')[:100]
            print(f'  {role}: {text}...')
else:
    print('No conversations file found')
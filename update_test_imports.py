#!/usr/bin/env python3
"""
Update all test file imports from memory. to advanced-memory
"""
from pathlib import Path
import re

project_root = Path(__file__).parent

# Files to update
test_files = [
    "backend/test_integration.py",
    "backend/test_mysql_connector.py", 
    "backend/test_mysql_integration.py",
    "backend/test_persistence.py",
    "backend/test_server_startup.py",
    "backend/test_simple_persistence.py",
    "backend/test_end_to_end.py"
]

import_header = """import sys
from pathlib import Path

# Add advanced-memory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "advanced-memory"))

"""

for file_path in test_files:
    full_path = project_root / file_path
    if not full_path.exists():
        continue
        
    print(f"Updating {file_path}...")
    
    with open(full_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already updated
    if 'advanced-memory' in content:
        print(f"  Already updated, skipping")
        continue
    
    # Replace imports
    content = re.sub(r'from memory\.hybrid import HybridMemory', 'from hybrid import HybridMemory', content)
    content = re.sub(r'from memory\.db import MemoryDB', 'from db import MemoryDB', content)
    content = re.sub(r'from memory\.db import AllieMemoryDB', 'from db import AllieMemoryDB', content)
    
    # Add import header after shebang/docstring if not already present
    lines = content.split('\n')
    insert_pos = 0
    
    # Skip shebang
    if lines and lines[0].startswith('#!'):
        insert_pos = 1
    
    # Skip docstring
    in_docstring = False
    for i in range(insert_pos, len(lines)):
        if '"""' in lines[i] or "'''" in lines[i]:
            if not in_docstring:
                in_docstring = True
            else:
                insert_pos = i + 1
                break
        elif not in_docstring and lines[i].strip() and not lines[i].strip().startswith('#'):
            insert_pos = i
            break
    
    # Insert path setup if not already there
    if 'sys.path.insert' not in content:
        lines.insert(insert_pos, import_header)
        content = '\n'.join(lines)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Updated")

print("\n✅ All test files updated")

"""Start a detached localhost chat server with persistent diagnostic logs."""
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

root = Path(__file__).resolve().parent
try:
    urllib.request.urlopen('http://127.0.0.1:7860/api/status', timeout=2)
    print('Chat server already running: http://127.0.0.1:7860')
except OSError:
    logs = root / 'results'
    logs.mkdir(exist_ok=True)
    with (logs / 'chat_server.log').open('ab') as output:
        flags = (subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
        process = subprocess.Popen([sys.executable, '-u', '-B', str(root / 'chat_server.py')],
            cwd=root, stdin=subprocess.DEVNULL, stdout=output, stderr=output,
            creationflags=flags, close_fds=True)
    print(f'Started chat server PID {process.pid}: http://127.0.0.1:7860')

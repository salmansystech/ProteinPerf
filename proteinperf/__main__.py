import sys
from proteinperf.cli import main

# Windows consoles and pipes may default to a codepage that can't print symbols like × or ⚠
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

main()

"""Put src/ on sys.path so scripts can `import eos`."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

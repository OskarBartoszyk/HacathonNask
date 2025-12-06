import anonbert
from anonbert import anonimze_file, anonimze_with_fakefill
from pathlib import Path

if __name__ == "__main__":
    input_path = Path("data/test.txt")
    anonimze_file(input_path)
    anonimze_with_fakefill(input_path)
from pathlib import Path


def load_prompt(filename: str) -> str:
    current_dir = Path(__file__).parent

    with open(current_dir / filename, 'r', encoding='utf-8') as file:
        return file.read()

# Generate prompts from cards

import ollama, re, textwrap
from pathlib import Path

MODEL = "gemma3:4b"

DIR = "../boozecube.mse-set/"
OUT = "prompts/"

FILTER_OUT = [
    "casting_cost:",
    "mse_version:",
    "time_created:",
    "time_modified:",
    "image:",
    "illustrator_brush:",
]

for file in Path(DIR).iterdir():
    if file.is_file() and file.name.startswith("card "):
        if Path(OUT, file.name).exists():
            continue
        card_text = file.read_text(encoding="utf-8")

        # Remove tags
        card_text = re.sub(r"</?[^>]+>", "", card_text)
        # Remove empty fields
        card_text = "\n".join(
            line
            for line in card_text.splitlines()[2:]
            if not line.strip().endswith(":")
            and not any(tag in line for tag in FILTER_OUT)
        )
        card_text = textwrap.dedent(card_text)
        card_text = "card:\n" + textwrap.indent(card_text, "  ")

        print("Processing card: " + file.name[5:])

        response = ollama.generate(
            model=MODEL,
            prompt="""
Generate a prompt for the image of the Magic The Gathering card described below. The prompt should be on a single line, describing the card's art, and should be in the format of a prompt for Stable Diffusion. The art may be detailed or stylish and simple, depending on the card's characteristics.\n\n"""
            + card_text,
            keep_alive="30m",
        )

        Path(OUT, file.name).write_text(response["response"], encoding="utf-8")

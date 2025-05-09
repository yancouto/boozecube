# Generate prompts from cards

import ollama, re, textwrap
from pathlib import Path

GPT = True
DIR = Path("../boozecube.mse-set/")

if GPT:
    OUT = Path("prompts_gpt/")
    from openai import OpenAI

    gpt_client = OpenAI()
else:
    OUT = Path("prompts/")
    gpt_client = None

FILTER_OUT = [
    "casting_cost:",
    "mse_version:",
    "time_created:",
    "time_modified:",
    "image:",
    "illustrator_brush:",
]

OUT.mkdir(parents=True, exist_ok=True)

for file in DIR.iterdir():
    if file.is_file() and file.name.startswith("card "):
        if (OUT / file.name).exists():
            continue
        card_text = file.read_text(encoding="utf-8")
        if "image: i" in card_text:
            # Skip cards with images
            continue

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
        if gpt_client is None:
            response = ollama.generate(
                model="gemma3:4b",
                prompt="""
    Generate a prompt for the image of the Magic The Gathering card described below. The prompt should be on a single line, describing the card's art, and should be in the format of a prompt for Stable Diffusion. The art may be detailed or stylish and simple, depending on the card's characteristics.\n\n"""
                + card_text,
                keep_alive="30m",
            )
            text = response["response"]
        else:
            response = gpt_client.responses.create(
                model="gpt-4.1",
                instructions="""
    Generate a prompt for the image of the Magic The Gathering card described. The prompt should be on a single line, describing the card's art, and should be in the format of a prompt for ChatGPT, without introduction. The art may be detailed or stylish and simple, depending on the card's characteristics.
        """,
                input=card_text,
                text={"format": {"type": "text"}},
                reasoning={},
                tools=[],
                temperature=1,
                max_output_tokens=2048,
                top_p=1,
                store=True,
            )
            text = response.output_text
        (OUT / file.name).write_text(text, encoding="utf-8")

from pathlib import Path


DIR = "../boozecube.mse-set/"
PROMPTS = "prompts/"
OUT = "cur_prompts/"

for file in Path(DIR).iterdir():
    if file.is_file() and file.name.startswith("card "):
        if Path(OUT, file.name).exists():
            continue
        card_text = file.read_text(encoding="utf-8")
        if "image: i" in card_text:
            continue

        Path(OUT, file.name).write_text(
            Path(PROMPTS, file.name).read_text(encoding="utf-8"), encoding="utf-8"
        )

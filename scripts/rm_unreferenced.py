from pathlib import Path

set_txt = Path("../boozecube.mse-set/set").read_text(encoding="utf-8")

for file in Path("../boozecube.mse-set/").iterdir():
    if (
        file.is_file()
        and file.name.startswith("card ")
        and (file.name + "\n") not in set_txt
    ):
        print("Deleting unreferenced file: " + file.name)
        file.unlink()

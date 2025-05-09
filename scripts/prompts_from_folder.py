# This was copied and edited from prompts_from_file.py and is used in
# WebUI to generate images from a folder of prompt files.

import copy
import random
import shlex
from pathlib import Path
import modules.scripts as scripts
import gradio as gr

from modules import sd_samplers, errors, sd_models
from modules.processing import Processed, process_images
from modules.shared import state


def process_model_tag(tag):
    info = sd_models.get_closet_checkpoint_match(tag)
    assert info is not None, f"Unknown checkpoint: {tag}"
    return info.name


def process_string_tag(tag):
    return tag


def process_int_tag(tag):
    return int(tag)


def process_float_tag(tag):
    return float(tag)


def process_boolean_tag(tag):
    return True if (tag == "true") else False


prompt_tags = {
    "sd_model": process_model_tag,
    "outpath_samples": process_string_tag,
    "outpath_grids": process_string_tag,
    "prompt_for_display": process_string_tag,
    "prompt": process_string_tag,
    "negative_prompt": process_string_tag,
    "styles": process_string_tag,
    "seed": process_int_tag,
    "subseed_strength": process_float_tag,
    "subseed": process_int_tag,
    "seed_resize_from_h": process_int_tag,
    "seed_resize_from_w": process_int_tag,
    "sampler_index": process_int_tag,
    "sampler_name": process_string_tag,
    "batch_size": process_int_tag,
    "n_iter": process_int_tag,
    "steps": process_int_tag,
    "cfg_scale": process_float_tag,
    "width": process_int_tag,
    "height": process_int_tag,
    "restore_faces": process_boolean_tag,
    "tiling": process_boolean_tag,
    "do_not_save_samples": process_boolean_tag,
    "do_not_save_grid": process_boolean_tag,
}


def cmdargs(line):
    args = shlex.split(line)
    pos = 0
    res = {}

    while pos < len(args):
        arg = args[pos]

        assert arg.startswith("--"), f'must start with "--": {arg}'
        assert pos + 1 < len(args), f"missing argument for command line option {arg}"

        tag = arg[2:]

        if tag == "prompt" or tag == "negative_prompt":
            pos += 1
            prompt = args[pos]
            pos += 1
            while pos < len(args) and not args[pos].startswith("--"):
                prompt += " "
                prompt += args[pos]
                pos += 1
            res[tag] = prompt
            continue

        func = prompt_tags.get(tag, None)
        assert func, f"unknown commandline option: {arg}"

        val = args[pos + 1]
        if tag == "sampler_name":
            val = sd_samplers.samplers_map.get(val.lower(), None)

        res[tag] = func(val)

        pos += 2

    return res


def load_prompts_from_directory(directory):
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {directory}")

    prompts = {}
    for file in directory_path.iterdir():
        if file.is_file():
            prompts[file.name] = file.read_text(encoding="utf-8").strip()
    return prompts


class Script(scripts.Script):
    def title(self):
        return "Prompts from folder"

    def ui(self, is_img2img):
        prompt_dir = gr.Textbox(
            label="Directory of prompt files",
            lines=1,
            elem_id=self.elem_id("prompt_dir"),
        )
        output_dir = gr.Textbox(
            label="Output directory for images",
            lines=1,
            elem_id=self.elem_id("output_dir"),
        )
        checkbox_iterate = gr.Checkbox(
            label="Iterate seed every line",
            value=False,
            elem_id=self.elem_id("checkbox_iterate"),
        )
        checkbox_iterate_batch = gr.Checkbox(
            label="Use same random seed for all lines",
            value=False,
            elem_id=self.elem_id("checkbox_iterate_batch"),
        )
        prompt_position = gr.Radio(
            ["start", "end"],
            label="Insert prompts at the",
            elem_id=self.elem_id("prompt_position"),
            value="start",
        )
        return [
            prompt_dir,
            output_dir,
            checkbox_iterate,
            checkbox_iterate_batch,
            prompt_position,
        ]

    def run(
        self,
        p,
        prompt_dir: str,
        output_dir: str,
        checkbox_iterate,
        checkbox_iterate_batch,
        prompt_position,
    ):
        prompts = load_prompts_from_directory(prompt_dir)

        p.do_not_save_grid = True

        job_count = 0
        jobs = []

        for filename, prompt in prompts.items():
            if "--" in prompt:
                try:
                    args = cmdargs(prompt)
                except Exception:
                    errors.report(
                        f"Error parsing file {filename} as commandline", exc_info=True
                    )
                    args = {"prompt": prompt}
            else:
                args = {"prompt": prompt}

            args["filename"] = filename  # Store the filename for later use
            job_count += args.get("n_iter", p.n_iter)
            jobs.append(args)

        print(f"Will process {len(prompts)} files in {job_count} jobs.")
        if (checkbox_iterate or checkbox_iterate_batch) and p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.job_count = job_count

        images = []
        all_prompts = []
        infotexts = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for args in jobs:
            state.job = f"{state.job_no + 1} out of {state.job_count}"

            copy_p = copy.copy(p)
            for k, v in args.items():
                if k == "sd_model":
                    copy_p.override_settings["sd_model_checkpoint"] = v
                elif k != "filename":  # Exclude filename from being set as an attribute
                    setattr(copy_p, k, v)

            if args.get("prompt") and p.prompt:
                if prompt_position == "start":
                    copy_p.prompt = args.get("prompt") + " " + p.prompt
                else:
                    copy_p.prompt = p.prompt + " " + args.get("prompt")

            if args.get("negative_prompt") and p.negative_prompt:
                if prompt_position == "start":
                    copy_p.negative_prompt = (
                        args.get("negative_prompt") + " " + p.negative_prompt
                    )
                else:
                    copy_p.negative_prompt = (
                        p.negative_prompt + " " + args.get("negative_prompt")
                    )

            proc = process_images(copy_p)
            for img in proc.images:
                img.save(output_path / f"{args['filename']}.png")
            images += proc.images

            if checkbox_iterate:
                p.seed = p.seed + (p.batch_size * p.n_iter)
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

        return Processed(
            p, images, p.seed, "", all_prompts=all_prompts, infotexts=infotexts
        )

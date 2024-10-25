"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            if isinstance(instruction, str) and isinstance(input, str):
                res = self.template["prompt_input"].format(
                    instruction=instruction, input=input
                )
            elif isinstance(instruction, list) and isinstance(input, list):
                res = []
                for single_instruction, single_input in zip(
                    instruction, input
                ):
                    res.append(
                        self.template["prompt_input"].format(
                            instruction=single_instruction, input=single_input
                        )
                    )
            else:
                raise Exception("instruction and input types are not matched")
        else:
            if isinstance(instruction, str):
                res = self.template["prompt_no_input"].format(
                    instruction=instruction
                )
            else:
                res = []
                for single_instruction, single_input in zip(
                    instruction, input
                ):
                    res.append(
                        self.template["prompt_input"].format(
                            instruction=single_instruction, input=single_input
                        )
                    )
        if label:
            if isinstance(label, str):
                res = f"{res}{label}"
            elif isinstance(label, list):
                res = [
                    f"{single_res}{single_label}"
                    for single_res, single_label in zip(res, label)
                ]
            else:
                raise Exception("check types of label")
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: Union[str, list]) -> Union[str, list]:
        if isinstance(output, str):
            return output.split(self.template["response_split"])[1].strip()
        elif isinstance(output, list):
            return [
                single_output.split(self.template["response_split"])[1].strip()
                for single_output in output
            ]

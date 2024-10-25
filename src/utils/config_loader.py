import datetime
import os

import yaml


class Config:
    def __init__(self, root=1, flatten=True, **entries):
        entries = self.process_keys(entries, flatten) if root == 1 else entries

        for key, value in entries.items():
            if isinstance(value, dict):
                ### This part is left for future consideration. Currently, it is meaningless due to the _flatten_nested_dict part.
                self.__dict__[key] = Config(root=0, flatten=flatten, **value)
            else:
                self.__dict__[key] = value

    def __repr__(self):
        return f"{self.__dict__}"
        # return f"Config({self.__dict__})"

    def _concat_path(self, entries):
        if "path" in entries and "root_dir" in entries:
            for sub_path in entries["path"]:
                if isinstance(entries["path"][sub_path], str):
                    entries["path"][sub_path] = os.path.join(
                        entries["root_dir"], entries["path"][sub_path]
                    )
                elif isinstance(entries["path"][sub_path], list):
                    entries["path"][sub_path] = os.path.join(
                        entries["root_dir"],
                        *sum(
                            [
                                x if isinstance(x, list) else [x]
                                for x in entries["path"][sub_path]
                            ],
                            [],
                        ),
                    )
        return entries

    def _flatten_nested_dict(self, entries):
        for k, v in list(entries.items()):
            if isinstance(v, dict):
                for v_k in v.keys():
                    assert (
                        v_k not in entries
                    ), f"{v_k} already exists in config!"
                    entries[v_k] = v[v_k]
                del entries[k]
        return entries

    def _setup_model_save_path(self, entries):
        if "output_dir" in entries:
            if (
                "num_masking_sets" in entries
                and entries["num_masking_sets"] != 1
            ):
                actual_epoch = (
                    entries["num_masking_sets"] * entries["num_epochs"]
                )
            else:
                actual_epoch = entries["num_epochs"]

            result_dir = f'{entries["data_subfolder_name"]}_epoch{actual_epoch}_lr{entries["learning_rate"]}_lora{entries["lora_r"]}-{entries["lora_alpha"]}_trainOnInputs{entries["train_on_inputs"]}'
            if "drop_ratio" in entries:
                result_dir = f'dropProp{entries["drop_ratio"]}_' + result_dir
            if "seed" in entries:
                result_dir = f'seed{entries["seed"]}_' + result_dir
            if "add_noise_to_output" in entries:
                result_dir = f'addNoise{entries["add_noise_to_output"][2]}_' + result_dir

            entries["output_dir"] = os.path.join(
                entries["output_dir"], result_dir
            )
        if "lora_weights" in entries and entries["lora_weights"] is None:
            entries["lora_weights"] = entries["output_dir"]

        return entries

    def process_keys(self, entries, flatten):
        entires = self._concat_path(entries)
        entries = self._flatten_nested_dict(entries) if flatten else entries
        entries = self._setup_model_save_path(entries)

        return entries

    @staticmethod
    def save_file(config, file_name="config"):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        assert getattr(
            config, "output_dir", None
        ), f"output_dir is missed\n{config}"
        os.makedirs(getattr(config, "output_dir"), exist_ok=True)

        save_path = os.path.join(
            getattr(config, "output_dir"), f"{file_name}_{current_time}.yaml"
        )
        with open(save_path, "w") as f:
            yaml.dump(config, f)

import random
import re

from torch.utils.data import Dataset
from datasets import Dataset as HFDataset


def count_decimal_places(num):
    num_str = str(num)
    return len(num_str.split('.')[1]) if len(num_str.split('.')) > 1 else 0

class BaseDataset(Dataset):
    def __init__(self, data, tokenizer, prompter, cfg):
        self.original_data = data
        self.data = self.original_data
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.cfg = cfg
        print("data sample:", self.data[0])
        self.data = self.data.map(self.generate_and_tokenize_prompt)
        ## is shuffle option not needed?

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=False,
            # truncation=True,
            # max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cfg.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        if not self.cfg.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = self.tokenize(
                user_prompt, add_eos_token=self.cfg.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.cfg.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return self.data[idx]


class DynamicTransformDataset(BaseDataset):
    def __init__(self, data, tokenizer, prompter, cfg):
        super().__init__(data, tokenizer, prompter, cfg)
        if hasattr(cfg, "drop_props_from_input"):
            self.pattern_list = cfg.drop_props_from_input
            self.transform = self._transform_input
        elif hasattr(cfg, "drop_props_from_instruction_output"):
            self.pattern_list = cfg.drop_props_from_instruction_output
            self.transform = self._transform_instruction_output
        elif hasattr(cfg, "add_noise_to_output"):
            self.pattern_list = cfg.add_noise_to_output
            self.transform = self._transform_output
            
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        # random.seed(int(self.current_epoch))
        self.transform()

    def _transform_input(self, replacement=""):
        self.processed_dataset = []
        for item in self.original_data:
            processed_item = DynamicTransformDataset.transform_input_single(
                item, self.pattern_list, replacement
            )
            self.processed_dataset.append(processed_item)
        self.data = list(
            map(self.generate_and_tokenize_prompt, self.processed_dataset)
        )
        print(
            "data sample:",
            self.processed_dataset[random.randint(0, 100) % len(self.data)],
        )

    def _transform_instruction_output(self, replacement=""):
        self.processed_dataset = []
        for item in self.original_data:
            processed_item = {k: v for k, v in item.items()}
            for pattern_in, pattern_out, drop_ratio in self.pattern_list:
                if (
                    re.search(pattern_out, processed_item["output"])
                    and random.random() < drop_ratio
                ):  # 50% 확률로 문자열 삭제
                    processed_item["instruction"] = re.sub(
                        pattern_in, replacement, processed_item["instruction"]
                    )
                    processed_item["output"] = re.sub(
                        pattern_out, replacement, processed_item["output"]
                    )
            self.processed_dataset.append(processed_item)
        self.data = list(
            map(self.generate_and_tokenize_prompt, self.processed_dataset)
        )
        print(
            "data sample:",
            self.processed_dataset[random.randint(0, 100) % len(self.data)],
        )

    def _transform_output(self, replacement=""):
        self.processed_dataset = []
        pattern_in, pattern_out, noise_ratio = self.pattern_list
        for item in self.original_data:
            processed_item = {k: v for k, v in item.items()}
            processed_item["output"] = re.sub(pattern_in, lambda match: pattern_out.format(str(round(random.gauss(float(match.group(1)), noise_ratio), count_decimal_places(match.group(1))))), processed_item["output"])
            self.processed_dataset.append(processed_item)
        self.data = list(
            map(self.generate_and_tokenize_prompt, self.processed_dataset)
        )
        print(
            "changed data sample:",
            self.processed_dataset[0],
        )

    @staticmethod
    def transform_input_single(item, pattern_list, replacement):
        processed_item = {k: v for k, v in item.items()}
        for pattern, drop_ratio in pattern_list:
            if (
                re.search(pattern, processed_item["input"])
                and random.random() < drop_ratio
            ):  # 50% 확률로 문자열 삭제
                processed_item["input"] = re.sub(
                    pattern, replacement, processed_item["input"]
                )
        return processed_item

    def __len__(self):
        return len(self.original_data)


class DPODynamicTransformDataset(DynamicTransformDataset):
    def __init__(self, data, tokenizer, prompter, cfg):
        self.column_names = data.column_names
        self.original_data = data
        self.data = self.original_data
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.cfg = cfg
        
        self.data = self.data.map(self.generate_prompt)
        print("column names:", self.data.column_names)       

        if hasattr(cfg, "drop_props_from_input"):
            self.pattern_list = cfg.drop_props_from_input
            self.transform = self._transform_input
        self.set_epoch(0)

        print("data sample:", self.data[0])
        self.data_dict = {k: [data[k] for data in self.data] for k in ["prompt", "chosen", "rejected"]}


    def set_epoch(self, epoch):
        self.current_epoch = epoch
        # random.seed(int(self.current_epoch))
        self.transform()

    def generate_prompt(self, data_point):
        prompt = self.prompter.generate_prompt(
            data_point["prompt"]["instruction"],
            data_point["prompt"]["input"],
        )
        chosen = data_point["chosen"]
        rejected = data_point["rejected"]
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    def _transform_input(self, replacement=""):
        self.processed_dataset = []
        for item in self.original_data:
            processed_item = DPODynamicTransformDataset.transform_input_single(
                item, self.pattern_list, replacement
            )
            self.processed_dataset.append(processed_item)
        self.data = list(
            map(self.generate_prompt, self.processed_dataset)
        )
        print(
            "data sample:",
            self.processed_dataset[random.randint(0, 100) % len(self.data)],
        )
        self.data = HFDataset.from_list(self.data)

    @staticmethod
    def transform_input_single(item, pattern_list, replacement):
        processed_item = {k: v for k, v in item.items()}
        for pattern, drop_ratio in pattern_list:
            if (
                re.search(pattern, processed_item["prompt"]["input"])
                and random.random() < drop_ratio
            ):  
                processed_item["prompt"]["input"] = re.sub(
                    pattern, replacement, processed_item["prompt"]["input"]
                )
        return processed_item

    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return self.data[idx]

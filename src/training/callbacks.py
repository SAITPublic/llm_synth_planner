from transformers import TrainerCallback
import torch

class UpdateEpochCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        if "train_dataset" in kwargs:
            kwargs["train_dataset"].set_epoch(state.epoch)


class GPUUsageLogger(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            self.print_gpu_memory()

    @staticmethod
    def print_gpu_memory():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            print(f"GPU {i}: Allocated memory: {allocated:.2f} GB, Reserved memory: {reserved:.2f} GB")

import os
import time
import torch
import logging
import subprocess


class AutoGPU():
    """ Automatically choose a GPU with enough free memory """
    def __init__(self):
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            self.gpu_list = list(map(int, visible_devices.split(',')))
        else:
            self.gpu_list = [i for i in range(torch.cuda.device_count())]
        self.free_memory = {i: self.query_free_memory(j) for i, j in enumerate(self.gpu_list)} # cuda:i -> memory of j-th GPU
        self.logger = logging.getLogger('my.AutoGPU')

    def update_free_memory(self):
        for i, j in enumerate(self.gpu_list):
            self.free_memory[i] = self.query_free_memory(j)

    def choice_gpu(self, memory_MB, interval=600, force=True):
        """ Choose a GPU with enough free memory
        Arguments:
        - memory_MB: int, the required memory in MB
        - interval: int, the interval (in second) to check free memory
        - force: bool, whether to wait until a GPU is available
        """
        waiting = False
        while True:
            for i, free_memory in self.free_memory.items():
                if free_memory < memory_MB: continue
                try:
                    device = f'cuda:{i}'
                    free_memory1 = self.query_free_memory(self.gpu_list[i])
                    fuck_cuda = torch.zeros(int(memory_MB), 1024, 256, dtype=torch.float32, device=device)
                    free_memory2 = self.query_free_memory(self.gpu_list[i])
                    if waiting:
                        self.logger.note(f"SubProcess[{os.getpid()}]: Choose GPU{self.gpu_list[i]} ({device}) with {memory_MB}MB ({free_memory1}MB -> {free_memory2}MB)")
                    else:
                        self.logger.info(f"SubProcess[{os.getpid()}]: Choose GPU{self.gpu_list[i]} ({device}) with {memory_MB}MB ({free_memory1}MB -> {free_memory2}MB)")
                    del fuck_cuda
                    return device
                except Exception as e:
                    torch.cuda.empty_cache()
                    continue
            else:
                if force:
                    if not waiting:
                        self.logger.warning(f"SubProcess[{os.getpid()}]: Waiting GPU...")
                        waiting = True
                    time.sleep(interval)
                    self.update_free_memory()
                else: # not force
                    self.logger.warning(f"SubProcess[{os.getpid()}]: No available GPU!")
                    return 'cpu'

    def query_free_memory(self, gpu_id):
        try:
            cmd = f"nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i {gpu_id}"
            return int(subprocess.check_output(cmd, shell=True).decode().strip().split("\n")[0])
        except Exception as e:
            self.logger.warning(f'Query CUDA (GPU{gpu_id}) Memory Failed! {e}')
            return 0


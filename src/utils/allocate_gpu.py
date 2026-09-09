import torch
import time

# 1. Define the exact target in Gigabytes
TARGET_GB = 4

# 2. Calculate exact number of elements
# 1 GB = 1024^3 bytes. 
# A float32 takes 4 bytes.
elements_per_gb = (1024**3) // 4
total_elements = TARGET_GB * elements_per_gb

print(f"Attempting to allocate exactly {TARGET_GB}GB of GPU memory...")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Check your PyTorch installation.")

device = torch.device("cuda:0")

# 3. Allocate the tensor directly on the GPU
# Using torch.empty is instantaneous, but we must fill it to force physical allocation
memory_block = torch.empty(total_elements, dtype=torch.float32, device=device)

# 4. Fill the tensor with dummy data so it isn't optimized away
memory_block.fill_(1.0)

# 5. Verify the allocation from PyTorch's perspective
allocated = torch.cuda.memory_allocated(device) / (1024**3)
print(f"Success! PyTorch reports {allocated:.2f} GB actively allocated in tensors.")

# 6. Keep the process alive to hold the memory
print("Holding memory in VRAM... Press Ctrl+C to release and exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nReleasing memory...")
    del memory_block
    torch.cuda.empty_cache()
import torch

checkpoint_1 = torch.load("work_dirs/denoised_training/checkpoint_10.pth")
checkpoint_2 = torch.load("work_dirs/denoised_training/checkpoint0010.pth")

for key in checkpoint_1:
    # Check if both values are tensors
    if isinstance(checkpoint_1[key], torch.Tensor) and isinstance(checkpoint_2[key], torch.Tensor):
        if not torch.equal(checkpoint_1[key], checkpoint_2[key]):  # Compare tensors
            print(f"Difference in {key}: Tensors differ")
    else:
        # For non-tensor values, use direct comparison
        if checkpoint_1[key] != checkpoint_2[key]:
            print(f"Difference in {key}: {checkpoint_1[key]} vs {checkpoint_2[key]}")

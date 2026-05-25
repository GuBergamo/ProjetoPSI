import torch

print("Torch version:")
print(torch.__version__)

print("\nCUDA available:")
print(torch.cuda.is_available())

print("\nGPU count:")
print(torch.cuda.device_count())

if torch.cuda.is_available():

    print("\nGPU name:")

    print(torch.cuda.get_device_name(0))

    device = torch.device("cuda")

else:

    print("\nUsing CPU")

    device = torch.device("cpu")

print("\nSelected device:")
print(device)
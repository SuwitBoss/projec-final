print("Attempting to import torch...")
try:
    import torch
    print(f"Successfully imported torch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        print(f"Torch CUDA version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Torch was likely compiled for CPU.")
except Exception as e:
    print(f"Error importing torch: {e}")
    import traceback
    traceback.print_exc()

print("\nAttempting to import tqdm...")
try:
    import tqdm
    print(f"Successfully imported tqdm version: {tqdm.__version__}")
except Exception as e:
    print(f"Error importing tqdm: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete.")

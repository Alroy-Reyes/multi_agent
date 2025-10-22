# Add to train_ppo.py to check
import psutil

print("\n" + "="*60)
print("SYSTEM MEMORY CHECK")
print("="*60)
total_ram = psutil.virtual_memory().total / (1024**3)
available_ram = psutil.virtual_memory().available / (1024**3)
used_percent = psutil.virtual_memory().percent

print(f"Total RAM: {total_ram:.2f} GB")
print(f"Available RAM: {available_ram:.2f} GB")
print(f"Used: {used_percent:.1f}%")

if available_ram < 4:
    print("\n⚠️ WARNING: Less than 4 GB available!")
    print("Recommend: Close other programs or use smaller batch size")
elif available_ram < 8:
    print("\n⚠️ Limited RAM: Use batch_size=256-384")
else:
    print("\n✓ Sufficient RAM for batch_size=512")

print("="*60 + "\n")
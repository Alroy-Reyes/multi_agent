# inspect_events.py
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# point this at your Ray results directory:
basedir = r"C:\Users\USER\ray_results\PPO_Timetabling"
# find all the events files
files = glob.glob(f"{basedir}/**/events.out.tfevents.*", recursive=True)
for f in files:
    print(f"\n=== {f} ===")
    ea = EventAccumulator(f,
        size_guidance={  # see everything
            EventAccumulator.SCALARS: 0,
            EventAccumulator.IMAGES: 0,
            EventAccumulator.HISTOGRAMS: 0,
            EventAccumulator.COMPRESSED_HISTOGRAMS: 0,
        })
    try:
        ea.Reload()  
        tags = ea.Tags().get("scalars", [])
        print("Scalar tags:", tags)
        if "episode_reward_mean" in tags:
            vals = ea.Scalars("episode_reward_mean")
            print("First 5 reward entries:", vals[:5])
    except Exception as e:
        print("  ⛔  Failed to read:", e)

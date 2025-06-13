import os, glob

basedir = os.path.expanduser("~/ray_results/PPO_Timetabling")
paths = glob.glob(f"{basedir}/**/events.out.tfevents.*", recursive=True)
if paths:
    print("Found event files:")
    for p in paths:
        print("  ", p, os.path.getsize(p), "bytes")
else:
    print("No event files found in", basedir)

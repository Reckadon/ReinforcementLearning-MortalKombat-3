import cv2
import os

def pngs_to_video(folder, output_path, fps=30):
    # Get sorted list of .png files
    files = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.endswith(".png")
    ])

    if len(files) == 0:
        print(f"[SKIP] No PNG files in {folder}")
        return

    # Read first frame
    frame = cv2.imread(files[0])
    height, width, _ = frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for img_path in files:
        img = cv2.imread(img_path)
        out.write(img)

    out.release()
    print(f"[OK] Saved video: {output_path}")


# ============================
# Process ALL episodes
# ============================
root_dir = "videos"

for episode_name in os.listdir(root_dir):
    episode_path = os.path.join(root_dir, episode_name)

    # Only folders like episode_000, episode_001...
    if os.path.isdir(episode_path) and episode_name.startswith("episode_"):
        output_video = os.path.join(root_dir, f"{episode_name}.mp4")
        print(f"Processing {episode_name}...")
        pngs_to_video(episode_path, output_video, fps=30)
print("All done!")
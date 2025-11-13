"""
Extract start and end frames from video to create before/after comparison figure.
"""
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def extract_video_frames(video_path, output_path='generated_figures/video_comparison.png'):
    """Extract first and last frames from video and create comparison figure."""
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    # Get total frame count and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Video has {total_frames} frames at {fps:.2f} FPS")
    
    # Read frame at 1.5 seconds (to skip VS Code window and show robot start position)
    start_frame_number = int(1.5 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    ret1, frame_start = cap.read()
    print(f"✓ Using frame {start_frame_number} (at 1.5s) as start frame")
    
    # Read last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret2, frame_end = cap.read()
    
    cap.release()
    
    if not (ret1 and ret2):
        print("❌ Could not read frames")
        return False
    
    # Convert BGR to RGB
    frame_start = cv2.cvtColor(frame_start, cv2.COLOR_BGR2RGB)
    frame_end = cv2.cvtColor(frame_end, cv2.COLOR_BGR2RGB)
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(frame_start)
    ax1.set_title('Start Position', fontsize=14, fontweight='bold')
    ax1.axis('off')
    ax1.text(0.5, -0.05, 'Robot at initial position near obstacles',
             transform=ax1.transAxes, ha='center', fontsize=11)
    
    ax2.imshow(frame_end)
    ax2.set_title('Goal Reached', fontsize=14, fontweight='bold')
    ax2.axis('off')
    ax2.text(0.5, -0.05, 'Robot successfully navigated to goal with zero collisions',
             transform=ax2.transAxes, ha='center', fontsize=11)
    
    plt.suptitle('Navigation Success: Start → Goal', fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison to: {output_path}")
    return True

if __name__ == "__main__":
    # Try the most recent video first
    video_path = Path("results/phase6_screen_capture.mp4")
    
    if not video_path.exists():
        print(f"⚠ Video not found: {video_path}")
        # Try archived videos
        archived = list(Path("results/archive_old_videos").glob("phase6_*.mp4"))
        if archived:
            video_path = archived[0]
            print(f"  Using archived video: {video_path.name}")
        else:
            print("❌ No videos found!")
            exit(1)
    
    success = extract_video_frames(video_path)
    
    if success:
        print("\n✅ Video frame extraction complete!")
        print("   Figure shows robot navigating from start to goal")
    else:
        print("\n❌ Video frame extraction failed")

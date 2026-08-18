"""Drawing utilities for match results."""
import json
import cv2
import matplotlib.pyplot as plt

def visualize_localization(json_output_path="sample_0001_A.json"):
    # Load detection metadata / predictions
    with open(json_output_path, "r") as f:
        data = json.load(f)

    # Read search and reference images
    search_img_path = f"sample_0001_A_search.png"
    ref_img_path = f"sample_0001_A_reference.png"

    search_img = cv2.imread(search_img_path)
    ref_img = cv2.imread(ref_img_path)

    # Convert BGR to RGB for Matplotlib
    search_img_rgb = cv2.cvtColor(search_img, cv2.COLOR_BGR2RGB)
    ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    # Extract target parameters from metadata
    center_x = data["target_x"]
    center_y = data["target_y"]
    box_size = data["target_size"]

    # Calculate bounding box coordinates
    top_left_x = int(center_x - box_size // 2)
    top_left_y = int(center_y - box_size // 2)
    bottom_right_x = int(center_x + box_size // 2)
    bottom_right_y = int(center_y + box_size // 2)

    # Draw bounding box (Green) and center point (Red)
    annotated_img = search_img_rgb.copy()
    cv2.rectangle(annotated_img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
    cv2.circle(annotated_img, (center_x, center_y), 3, (255, 0, 0), -1)

    # Plot side-by-side comparison for demonstration
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(ref_img_rgb)
    axes[0].set_title("Reference Template")
    axes[0].axis("off")

    axes[1].imshow(search_img_rgb)
    axes[1].set_title("Search Image (Raw)")
    axes[1].axis("off")

    axes[2].imshow(annotated_img)
    axes[2].set_title(f"Localized Target ({center_x}, {center_y})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("demo_visualization_result.png", dpi=300)
    print("Saved annotated output image to: demo_visualization_result.png")
    plt.show()

if __name__ == "__main__":
    visualize_localization()
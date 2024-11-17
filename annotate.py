import typer
import torch
from ultralytics import YOLO
import os

app = typer.Typer()

@app.command()
def predict(
    unlabeled_dir: str = typer.Argument(..., help="Directory containing unlabeled images."),
    class_names: str = typer.Option("this_spot", "-c", "--classes", help="Comma-separated list of class names."),
    model_path: str = typer.Option("trained_model/refined/best.pt", "-m", "--model", help="Path to the trained YOLO model."),
    labels_dir: str = typer.Option(None, "-l", "--labels-dir", help="Directory to save annotated images (defaults to unlabeled_dir)"),
):
    """
    Predicts bounding boxes on unlabeled images and generates YOLO annotation files.
    """
    
    labels_dir = labels_dir or unlabeled_dir
    print(f"Saving annotated images to: {labels_dir}")  # Print for debugging
    # Load the trained model
    model = YOLO(model_path)

    # Get a list of unlabeled images
    unlabeled_images = [
        os.path.join(unlabeled_dir, f)
        for f in os.listdir(unlabeled_dir)
        if f.endswith((".jpg", ".jpeg", ".png"))
    ]
    
    # Create classes.txt
    os.makedirs(labels_dir, exist_ok=True)
    with open(os.path.join(labels_dir, "classes.txt"), "w") as f:
        for class_name in class_names.split(","):
            f.write(class_name + "\n")

    # Run inference on unlabeled images and generate YOLO annotation files
    for image_path in unlabeled_images:
        results = model(image_path)
        detections = results[0].boxes.data
        
         # Generate YOLO annotation file (in the labels directory)
        filename = os.path.basename(image_path)
        annotation_file = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")

        print(f"Generating annotation file: {annotation_file}")  # Print for debugging
        with open(annotation_file, "w") as f:
            for det in detections:
                class_id = int(det[5])
                # Convert tensor values to Python numbers before rounding
                x_center = round(float((det[0] + det[2]) / 2 / results[0].orig_shape[1]), 6)
                y_center = round(float((det[1] + det[3]) / 2 / results[0].orig_shape[0]), 6)
                width = round(float((det[2] - det[0]) / results[0].orig_shape[1]), 6)
                height = round(float((det[3] - det[1]) / results[0].orig_shape[0]), 6)

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")



if __name__ == "__main__":
    app()
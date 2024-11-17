import typer
import subprocess
import torch
from ultralytics import YOLO
import os

app = typer.Typer()

# Define the Python executable path
PYTHON_EXECUTABLE = r"d:\coding\bots\ml\env\scripts\python.exe"


@app.command(short_help="Detailed training steps")
def train_steps(
    ctx: typer.Context,
):
    """
    Displays detailed steps for training a YOLO model.
    """
    if ctx.args:
        typer.echo(f"Unsupported arguments: {ctx.args}")
        raise typer.Exit(code=1)

    typer.echo("Detailed steps for training a YOLO model:")
    typer.echo("1. **Capture screenshots:** Use the `capture` command to capture screenshots for your training data.")
    typer.echo("   Example: `python ml.py capture -o ml_training/my_dataset -p my_prefix -k ctrl+shift+p`")
    typer.echo("2. **Generate annotations:** Use the `annotate` command to automatically annotate the captured images.")
    typer.echo("   Example: `python ml.py annotate ml_training/my_dataset -m trained_model/refined/best.pt -l ml_training/my_dataset_labels`")
    typer.echo("3. **Verify annotations:** Use the `label` command to launch labelimg and manually verify or correct the annotations.")
    typer.echo("   Example: `python ml.py label`")
    typer.echo("4. **Train the model:** Use the `train` command to train a YOLO model on the annotated dataset.")
    typer.echo("   Example: `python ml.py train ml_training/my_dataset_labels -m yolov5su.pt -e 30 -b 8`")
    typer.echo("5. **Locate the trained model:** After training, the best.pt model file will be saved in the `runs/detect/trainXX/weights` directory, where XX is the training run number.")
    typer.echo("   Copy this file to your desired location for future use.")

@app.command(short_help="Capture screenshots")
def capture(
    ctx: typer.Context,
    output_dir: str = typer.Option("ml_training/images", "-o", "--output-dir", help="Directory to save screenshots"),
    prefix: str = typer.Option("speech_bubble", "-p", "--prefix", help="Filename prefix for screenshots"),
    hotkey: str = typer.Option("f9", "-k", "--hotkey", help="Hotkey to trigger screenshot capture")
):
    """
    Capture screenshots with custom settings.
    """
    if ctx.args:
        typer.echo(f"Unsupported arguments: {ctx.args}")
        raise typer.Exit(code=1)
    try:
        # Construct the command with arguments
        command = [
            PYTHON_EXECUTABLE,
            "image_capture.py",
            "--output-dir", output_dir,
            "--prefix", prefix,
            "--hotkey", hotkey
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error capturing screenshots: {e}")

@app.command(short_help="Launch labelimg")
def label(ctx: typer.Context):
    """
    Launch labelimg for image annotation.
    """
    if ctx.args:
        typer.echo(f"Unsupported arguments: {ctx.args}")
        raise typer.Exit(code=1)
    try:
        # Construct the command with the full path to labelimg
        command = [
            PYTHON_EXECUTABLE, 
            "env\\Lib\\site-packages\\labelImg\\labelImg.py"
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error launching labelimg: {e}")
  
@app.command(short_help="Predict and annotate images")
def annotate(
    ctx: typer.Context,
    unlabeled_dir: str = typer.Argument(
        ..., help="Directory containing unlabeled images."
    ),
    class_names: str = typer.Option(
        "this_spot", "-c", "--classes", help="Comma-separated list of class names."
    ),
    model_path: str = typer.Option(
        "trained_model/refined/best.pt",
        "-m",
        "--model",
        help="Path to the trained YOLO model.",
    ),
    labels_dir: str = typer.Option(
        "",
        "-l",
        "--labels-dir",
        help="Directory to save annotated images (defaults to unlabeled_dir)",
    ),
):
    """
    Predicts bounding boxes on unlabeled images and generates YOLO annotation files.
    """
    if ctx.args:
        typer.echo(f"Unsupported arguments: {ctx.args}")
        raise typer.Exit(code=1)
    try:
        # Construct the command with arguments
        command = [
            PYTHON_EXECUTABLE,
            "annotate.py",  # Call the annotate.py script
            unlabeled_dir,
            "--model",
            model_path,
            "--classes",
            class_names,
            "--labels-dir",
            labels_dir,
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error during prediction and annotation: {e}")


@app.command(short_help="Train YOLO model")
def train(
    ctx: typer.Context,
    image_dir: str = typer.Argument(
        ...,
        help="Directory containing training images and their annotation files.",
    ),
    model_path: str = typer.Option(
        "yolov5su.pt", "-m", "--model", help="Path to the model to be trained."
    ),
    epochs: int = typer.Option(20, "-e", "--epochs", help="Number of training epochs."),
    batch_size: int = typer.Option(
        3, "-b", "--batch-size", help="Batch size for training."
    ),
):
    """
    Trains a YOLO model on the given dataset.
    """
    if ctx.args:
        typer.echo(f"Unsupported arguments: {ctx.args}")
        raise typer.Exit(code=1)
    try:
        # Construct the command with arguments
        command = [
            PYTHON_EXECUTABLE,
            "train.py",  # Call the train.py script
            image_dir,
            "--model",
            model_path,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ]
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"Error during training: {e}")


        
if __name__ == "__main__":
    app()
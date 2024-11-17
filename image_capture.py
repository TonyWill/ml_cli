import keyboard
import pyautogui
import os
import typer

app = typer.Typer()

@app.command()
def capture(
    output_dir: str = typer.Option("ml_training/speech_bubbles", "-o", "--output-dir", help="Directory to save screenshots"),
    prefix: str = typer.Option("speech_bubble", "-p", "--prefix", help="Filename prefix for screenshots"),
    hotkey: str = typer.Option("f9", "-k", "--hotkey", help="Hotkey to trigger screenshot capture")
):
    """
    Capture screenshots with custom settings.
    """
    file_number = 1  # Initialize file_number here

    def capture_screenshot():
        """Captures a screenshot and saves it with the specified naming convention."""
        nonlocal file_number  # Access the nonlocal file_number

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Capture and save the screenshot
        screenshot = pyautogui.screenshot()
        filepath = os.path.join(output_dir, f"{prefix}_{file_number:03d}.png")
        screenshot.save(filepath)
        typer.echo(f"Screenshot saved: {filepath}")
        file_number += 1  # Increment file_number after saving


    # Register the hotkey
    keyboard.add_hotkey(hotkey, capture_screenshot)
    typer.echo(f"Press '{hotkey}' to capture screenshots. Press 'esc' to exit.")
    keyboard.wait("esc")

if __name__ == "__main__":
    app()
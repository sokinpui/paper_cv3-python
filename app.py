import argparse
import sys

import torch

from ui import create_ui


def run_app(input_dir):
    """The main application entry point."""
    # server_name="0.0.0.0" makes it accessible from external IP (SSH tunnel/remote)
    demo = create_ui(input_dir)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


def run_reloader():
    """
    A wrapper that runs the application in a subprocess and watches for
    file changes, restarting the app when a .py file is modified.
    """
    import subprocess
    import time

    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print(
            "Error: 'watchdog' is required for --reload. Please install it: pip install watchdog",
            file=sys.stderr,
        )
        sys.exit(1)

    class CodeChangeHandler(FileSystemEventHandler):
        def __init__(self):
            self.process = None
            self.start_process()

        def start_process(self):
            if self.process:
                self.process.terminate()
                self.process.wait()

            command = [sys.executable] + [arg for arg in sys.argv if arg != "--reload"]
            self.process = subprocess.Popen(command)

        def on_modified(self, event):
            if event.src_path.endswith(".py"):
                print(f"Detected change in {event.src_path}, reloading...")
                self.start_process()

    print("Starting server with auto-reload enabled...")
    handler = CodeChangeHandler()
    observer = Observer()
    observer.schedule(handler, path=".", recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping reloader...")
        if handler.process:
            handler.process.terminate()
            handler.process.wait()
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Image Anomaly Detection Web UI")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        help="Directory containing images to list in UI",
        default=None,
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reloading on file changes (for development).",
    )

    args = parser.parse_args()

    if args.reload:
        run_reloader()
    else:
        run_app(args.input_dir)

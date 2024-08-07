import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import whisperx
import torch
from tqdm import tqdm
import ollama

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
)


class MeetingNotesHandler(FileSystemEventHandler):
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".m4a"):
            logging.info(f"New M4A file detected: {event.src_path}")
            self.transcribe_audio(event.src_path)

    def process_directory(self):
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".m4a"):
                filepath = os.path.join(self.directory_path, filename)
                txt_path = filepath.replace(".m4a", ".txt")
                if not os.path.exists(txt_path):
                    logging.info(f"Processing existing file: {filepath}")
                    self.transcribe_audio(filepath)

    def transcribe_audio(self, audio_file_path):
        output_file_path = audio_file_path.replace(".m4a", ".txt")
        print("Loading the model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Using device: {device}")

        # Load the model
        model = whisperx.load_model(
            "medium",
            device,
            compute_type=compute_type,
            asr_options={
                "max_new_tokens": 128,
                "clip_timestamps": "0",
                "hallucination_silence_threshold": 0,
                "hotwords": [],
            },
            language="en",
        )

        print(f"Transcribing audio file: {audio_file_path}")

        # Start time
        start_time = time.time()

        # Transcribe the audio
        result = model.transcribe(audio_file_path, batch_size=16, chunk_size=30)

        print("Transcription complete. Aligning words...")

        # If you want word-level timestamps
        alignment_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], alignment_model, metadata, audio_file_path, device
        )

        # Process word-level timestamps and save to file
        with open(output_file_path, "w") as f:
            if "word_segments" in result:
                for segment in tqdm(result["word_segments"], desc="Processing words"):
                    word = segment.get("word", "").strip()
                    if word:
                        f.write(f"{word} ")
            else:
                print(
                    "No 'word_segments' found in the result. Available keys:",
                    result.keys(),
                )

        # Calculate and print the total processing time
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Transcription saved to {output_file_path}")

        # Generate summary
        with open(output_file_path, "r") as f:
            text = f.read()
        summary = self.generate_summary(text)
        summary_file_path = output_file_path.replace(".txt", "_summary.txt")
        with open(summary_file_path, "w") as f:
            f.write(summary)
        print(f"Summary saved to {summary_file_path}")

    def generate_summary(self, text):
        if not text.strip():
            return "Error: No text available to summarize."

        try:
            prompt = f"""Please summarize the following meeting notes in a concise manner, highlighting the key points and action items:

    {text}

    Summary:"""

            response = ollama.chat(
                model="llama3",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            summary = response["message"]["content"].strip()
            return summary
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")
            return f"Error: Unable to generate summary. Reason: {str(e)}"


if __name__ == "__main__":
    path = "/Users/sws/Desktop/meeting-notes/audio"  # Adjust this path

    logging.info(f"Monitoring directory: {path}")
    for file in os.listdir(path):
        logging.info(f"Current file in directory: {file}")

    event_handler = MeetingNotesHandler(path)

    # Process existing files
    event_handler.process_directory()

    # Set up observer for new files
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    try:
        logging.info("File monitoring started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Stopping file monitoring...")
        observer.stop()
    observer.join()

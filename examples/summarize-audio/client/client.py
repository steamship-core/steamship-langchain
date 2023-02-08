import base64
import time

from steamship import File, RuntimeEnvironments, Steamship, Task, TaskState, check_environment
from termcolor import colored


def main():
    # This helper provides runtime API key prompting, etc.
    check_environment(RuntimeEnvironments.LOCALHOST)

    churchill_yt_url = "https://www.youtube.com/watch?v=MkTw3_PmKtc"
    with Steamship.temporary_workspace() as client:
        api = client.use("test-summarize-audio")
        yt_importer = client.use_plugin("youtube-file-importer")

        print(colored(f"Importing {churchill_yt_url}...", "blue"), end="", flush=True)

        import_task = File.create_with_plugin(
            client=client, plugin_instance=yt_importer.handle, url=churchill_yt_url
        )
        import_task.wait()
        audio_file = import_task.output
        print(colored("Done.", "blue"), flush=True)

        print(
            colored(
                "Requesting transcription (via Whisper) and summarization (via GPT). This may take several minutes. "
                "Please be patient.",
                "blue",
            ),
            flush=True,
        )

        summarize_task_response = api.invoke("/summarize_audio_file", file_handle=audio_file.handle)
        summarize_task = Task(client=client, **summarize_task_response)

        # instead of this loop, we could also use `summarize_task.wait()`
        # but that would not make for a nice user experience here, so instead we poll and print to screen.
        print(colored("Progress: ", "blue"), end="", flush=True)
        while summarize_task.state not in [TaskState.succeeded, TaskState.failed]:
            print("🚢", end="", flush=True)
            time.sleep(3)
            summarize_task.refresh()

        print(colored(f"\nTask {summarize_task.state}.", "blue"))

        if summarize_task.state == TaskState.succeeded:
            summary = base64.b64decode(summarize_task.output).decode("utf-8")
            print(colored("Summary:\n", "blue"), f"{summary.strip()}")


if __name__ == "__main__":
    main()

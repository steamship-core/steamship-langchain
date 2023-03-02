"""Load all the content from a directory."""
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel
from steamship import File, Steamship, SteamshipError

from steamship_langchain.file_loaders.base import BaseFileLoader


class DirectoryLoader(BaseModel):
    """Loading logic for loading documents from a directory."""

    client: Steamship
    "Provides Steamship workspace-scoping for Files"

    file_loader: BaseFileLoader
    "Selects which Loader to use for the files in the directory."

    skip_dot_files_and_dirs: bool = True
    "By default, the directory loader will ignore dot files. This means `.git/`, etc., will be ignored."

    skip_images: bool = True
    "By default, the directory loader will ignore image files."

    ignore_failures: bool = False
    "By default, the directory loader will fail if any individual file fails to load. Set to True to ignore individual failures."

    def load(
        self, path: str, glob: str = "**/*", metadata: Optional[Dict[str, str]] = None
    ) -> List[File]:
        """Import all files in directory matching the glob expression into Steamship."""
        p = Path(path)
        files = []
        if self.skip_dot_files_and_dirs:
            visible_files = filter(
                lambda dir_path: not any((part for part in dir_path.parts if part.startswith("."))),
                p.glob(glob),
            )
        else:
            visible_files = p.glob(glob)
        for f in visible_files:
            if f.is_dir():
                continue
            if self.skip_images and (f.name.endswith(".png") or f.name.endswith(".jpg")):
                continue
            if f.is_file():
                try:
                    file_list = self.file_loader.load(str(f.absolute()), metadata)
                    files.extend(file_list)
                except SteamshipError as e:
                    if not self.ignore_failures:
                        raise e
        return files

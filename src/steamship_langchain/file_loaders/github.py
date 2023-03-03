"""Import files from a GitHub repository into Steamship workspace."""
import tempfile
from pathlib import PurePath
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from steamship import File, Steamship, Tag
from steamship.data import TagKind, TagValueKey
from steamship.data.tags.tag_constants import ProvenanceTag

from steamship_langchain.file_loaders.directory import DirectoryLoader
from steamship_langchain.file_loaders.text import TextFileLoader


class GitHubRepositoryLoader(BaseModel):
    """Load a GitHub repository's files into a Steamship workspace.

    Creates new Files by uploading the content of repo files. File tags for identifying the source files and the
    time of import are automatically added, as well as for any custom metadata that is provided.
    This enables query-based retrieval for downstream processing.

    NOTE: Requires install of GitPython via `pip install GitPython`
    """

    client: Steamship
    "Provides Steamship workspace-scoping for File loading."

    repository_path: str
    "The Github repo path (part after https://github.com). Typically <org-name>/<repo-name>. Example: steamship-core/steamship-langchain"

    branch_or_tag: str
    "The ref to checkout and load." "Example: v2.0.1"

    glob: str = Field(
        default="**/*",
        description="Unix-style pathname expansion pattern. Use this to select which "
        "files will be loaded into Steamship.",
    )

    ignore_failures: bool = False
    "By default, the directory loader will fail if any individual file fails to load. Set to True to ignore individual failures."

    def __init__(
        self,
        repository_path: str,
        branch_or_tag: str = "main",
        glob: str = "**/*",
        ignore_failures=False,
        **kwargs,
    ):
        """Initialize the loader with Steamship workspace and GitHub repo URL.

        :param client: Steamship client for a workspace
        :param repository_url: full github repository url
        :param branch_or_tag: brnach or tag to checkout prior to loading
        :param glob: provides the ability to filter the repo files to select only certain files for inclusion
        :raises ValueError: if `gitpython` package has not been installed.
        """
        super().__init__(
            repository_path=repository_path,
            branch_or_tag=branch_or_tag,
            glob=glob,
            ignore_failures=ignore_failures,
            **kwargs,
        )
        try:
            import git  # noqa: F401
        except ImportError:
            raise ValueError(
                "failed to `import git`. Please install it via `pip install GitPython`"
            )

    def load(self, metadata: Optional[Dict[str, Any]] = None) -> List[File]:
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            from git import Repo

            repo_url = f"https://github.com/{self.repository_path}.git"
            repo = Repo.clone_from(url=repo_url, to_path=tmp_dir_name)
            git_cli = repo.git
            git_cli.checkout(self.branch_or_tag)

            dir_loader = DirectoryLoader(
                client=self.client,
                file_loader=TextFileLoader(client=self.client),
                ignore_failures=self.ignore_failures,
            )
            files = dir_loader.load(tmp_dir_name, glob=self.glob, metadata=metadata)

            # remove the file tags, replace with a proper URL tag
            for f in files:
                for t in f.tags:
                    if t.kind == TagKind.PROVENANCE and t.name == ProvenanceTag.FILE:
                        file_path = t.value.get(TagValueKey.STRING_VALUE, "")
                        full_path = PurePath(file_path)
                        local_path = full_path.relative_to(tmp_dir_name)
                        url = f"https://github.com/{self.repository_path}/blob/{self.branch_or_tag}/{local_path}"
                        Tag.create(
                            client=self.client,
                            file_id=f.id,
                            kind=TagKind.PROVENANCE,
                            name=ProvenanceTag.URL,
                            value={TagValueKey.STRING_VALUE: url},
                        )
                        t.delete()

            return files

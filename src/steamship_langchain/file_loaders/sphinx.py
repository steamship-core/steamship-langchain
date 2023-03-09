import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from steamship import Block, File, MimeTypes, Steamship, SteamshipError, Tag
from steamship.data import TagKind, TagValueKey
from steamship.data.tags.tag_constants import ProvenanceTag

from steamship_langchain.file_loaders import BaseFileLoader


def generate_url_tags(client: Steamship, url: str, metadata: Optional[Dict[str, str]] = None):
    tags = [
        Tag(
            client=client,
            kind=TagKind.TIMESTAMP,
            name="timestamp",
            value={"timestamp": datetime.now().isoformat()},
        ),
        Tag(
            client=client,
            kind=TagKind.PROVENANCE,
            name=ProvenanceTag.URL,
            value={TagValueKey.STRING_VALUE: url},
        ),
    ]
    if metadata:
        tags.append(Tag(client=client, kind="metadata", name="metadata", value=metadata))
    return tags


def _sanitize_text(text: str) -> str:
    """Replace known problematic characters in text with underscores.

    NB: This set is probably woefully incomplete, but should, at the least,
    prevent *known* failures.
    """
    return text.translate(
        str.maketrans(
            {
                "$": "_",
                "%": "_",
            }
        )
    )


class SphinxSiteLoaderBase(BaseFileLoader):
    """Load sphinx websites into Steamship.

    Note:
        Requires install of BeautifulSoup, via: `pip install bs4`
    """

    tag_name: str
    "Name of html tag to use for imports. Example: section"

    tag_attributes: Dict[str, str] = {}
    "Identifying attributes for the tag. Example: {'id': 'main-content'}"

    scheme: str = "https://"
    "The URL scheme to use when creating provenance tags"

    use_tag_id_in_provenance: bool = True
    "Controls appending section tags to provenance URLs (a la: #some-header-in-doc)"

    ignore_failures: bool = False
    "By default, the loader will fail if any individual file fails to load. Set to True to ignore individual failures."

    sanitize_files: bool = False
    "Some files may have content that causes Steamship to reject raw upload. Set to True to remove known problematic characters."

    def load(  # noqa: C901
        self, path: str, metadata: Optional[Dict[str, str]] = None
    ) -> List[File]:
        """Load documents."""
        from bs4 import BeautifulSoup

        def _clean_data(data: str) -> List[Tuple[str, str]]:
            soup = BeautifulSoup(data, "html.parser")
            doc_tags = soup.find_all(self.tag_name, self.tag_attributes)

            if len(doc_tags) == 0:
                return []

            return [
                (t.get_text(separator=" ", strip=True), t.get("id", "")) for t in doc_tags if t.text
            ]

        files = []
        for p in Path(path).rglob("*.html"):
            if p.is_dir():
                continue
            with p.open() as f:
                logging.info(f"loading: {p}")
                base_url = f"{self.scheme}{p.relative_to(path)}"
                texts = _clean_data(f.read())

                for text, section_id in texts:
                    if self.sanitize_files:
                        text = _sanitize_text(text)
                    url = base_url
                    if len(section_id) > 0 and self.use_tag_id_in_provenance:
                        url = f"{base_url}#{section_id}"

                    logging.debug(f"setting provenance to: {url}")
                    tags = generate_url_tags(client=self.client, url=url, metadata=metadata)

                    try:
                        f = File.create(
                            client=self.client,
                            mime_type=MimeTypes.TXT,
                            blocks=[Block(text=text)],
                            tags=tags,
                        )
                        files.append(f)
                    except SteamshipError as e:
                        if not self.ignore_failures:
                            raise e

        return files


class SphinxSiteLoader(SphinxSiteLoaderBase):
    """Load entire articles into an index at a time."""

    tag_name: str = "article"
    tag_attributes: Dict[str, str] = {"class": "bd-article", "role": "main"}
    use_tag_id_in_provenance: bool = False


class SphinxSiteSectionLoader(SphinxSiteLoaderBase):
    """Load articles section by section, linking to individual section headers in provenance."""

    tag_name: str = "section"


class ReadTheDocsLoader(SphinxSiteLoaderBase):
    """Load entire articles into an index, using ReadTheDos style main-content tags."""

    tag_name: str = "main"
    tag_attributes: Dict[str, str] = {"id": "main-content"}
    use_tag_id_in_provenance: bool = False

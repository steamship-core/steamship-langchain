import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from steamship import Block, File, MimeTypes, Steamship, Tag
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


class SphinxSiteLoaderBase(BaseFileLoader):
    """Load sphinx websites into Steamship.

    Note:
        Requires install of BeautifulSoup, via:
        `pip install bs4`
    """

    tag_name: str
    "Name of html tag to use for imports"

    tag_attributes: Dict[str, str] = {}
    "Identifying attributes for the tag"

    scheme: str = "https://"

    use_section_id_in_provenance: bool = True

    def load(self, path: str, metadata: Optional[Dict[str, str]] = None) -> List[File]:
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
            with open(p) as f:
                logging.info(f"loading: {p}")
                base_url = f"{self.scheme}{p.relative_to(path)}"
                texts = _clean_data(f.read())

                for text, section_id in texts:
                    url = base_url
                    if len(section_id) > 0 and self.use_section_id_in_provenance:
                        url = f"{base_url}#{section_id}"

                    logging.debug(f"setting provenance to: {url}")
                    tags = generate_url_tags(client=self.client, url=url, metadata=metadata)

                    files.append(
                        File.create(
                            client=self.client,
                            mime_type=MimeTypes.TXT,
                            blocks=[Block(text=text)],
                            tags=tags,
                        )
                    )

        return files


class SphinxSiteLoader(SphinxSiteLoaderBase):
    """Load entire articles into an index at a time."""

    tag_name: str = "article"
    tag_attributes: Dict[str, str] = {"class": "bd-article", "role": "main"}
    use_section_id_in_provenance: bool = False


class SphinxSiteSectionLoader(SphinxSiteLoaderBase):
    """Load articles section by section, linking to individual section headers in provenance."""

    tag_name: str = "section"


class ReadTheDocsLoader(SphinxSiteLoaderBase):
    """Load entire articles into an index, using ReadTheDos style main-content tags."""

    tag_name: str = "main"
    tag_attributes: Dict[str, str] = {"id": "main-content"}
    use_section_id_in_provenance: bool = False

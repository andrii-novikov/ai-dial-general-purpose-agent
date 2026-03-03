import io
from pathlib import Path

import pandas as pd
import pdfplumber
from aidial_client import Dial
from bs4 import BeautifulSoup


class DialFileContentExtractor:
    def __init__(self, endpoint: str, api_key: str):
        # TODO:
        # Set Dial client with endpoint as base_url and api_key
        self._client = Dial(base_url=endpoint, api_key=api_key)

    def extract_text(self, file_url: str) -> str:
        # TODO:
        # 1. Download with Dial client file by `file_url` (files -> download)
        response = self._client.files.download(url=file_url)
        # 2. Get downloaded file name and content
        filename = response.filename
        file_content = response.get_content()
        # 3. Get file extension, use for this `Path(filename).suffix.lower()`
        file_extension = Path(filename).suffix.lower()
        # 4. Call `__extract_text` and return its result
        return self.__extract_text(
            file_content=file_content, file_extension=file_extension, filename=filename
        )

    def __extract_text(
        self, file_content: bytes, file_extension: str, filename: str
    ) -> str:
        """Extract text content based on file type."""
        # TODO:
        # Wrap in `try-except` block:
        # try:
        #   1. if `file_extension` is '.txt' then return `file_content.decode('utf-8', errors='ignore')`
        try:
            if file_extension == ".txt":
                return file_content.decode("utf-8", errors="ignore")
            elif file_extension == ".pdf":
                #   2. if `file_extension` is '.pdf' then:
                #       - load it with `io.BytesIO(file_content)`
                #       - with pdfplumber.open PDF files bites
                #       - iterate through created pages adn create array with extracted page text
                #       - return it joined with `\n`
                pdf_bytes = io.BytesIO(file_content)
                with pdfplumber.open(pdf_bytes) as pdf:
                    pages = pdf.pages
                    text = "\n".join([page.extract_text() for page in pages])
                    return text
            elif file_extension == ".csv":
                #   3. if `file_extension` is '.csv' then:
                #       - decode `file_content` with encoding 'utf-8' and errors='ignore'
                #       - create csv buffer from `io.StringIO(decoded_text_content)`
                #       - read csv with pandas (pd) as dataframe
                #       - return dataframe to markdown (index=False)
                decoded_text_content = file_content.decode("utf-8", errors="ignore")
                csv_buffer = io.StringIO(decoded_text_content)
                df = pd.read_csv(csv_buffer)
                return df.to_markdown(index=False) or ""
            elif file_extension in [".html", ".htm"]:
                #   4. if `file_extension` is in ['.html', '.htm'] then:
                #       - decode `file_content` with encoding 'utf-8' and errors='ignore'
                #       - create BeautifulSoup with decoded html content, features set as 'html.parser' as `soup`
                #       - remove script and style elements: iterate through `soup(["script", "style"])` and `decompose` those scripts
                #       - return `soup.get_text(separator='\n', strip=True)`
                decoded_text_content = file_content.decode("utf-8", errors="ignore")
                soup = BeautifulSoup(decoded_text_content, features="html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator="\n", strip=True)
            else:
                #   5. otherwise return it as decoded `file_content` with encoding 'utf-8' and errors='ignore'
                return file_content.decode("utf-8", errors="ignore")

        except Exception as e:
            print(e)
            return ""

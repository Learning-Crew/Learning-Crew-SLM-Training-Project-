
# pip install huggingface_hub

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from configs.tokenizer_config import PretrainDatasetConfig

DATA_DIR = Path(__file__).resolve().parent
REPO_ID = "jsuhong24/Learning-Crew-Dataset"

config = PretrainDatasetConfig()

def load_data(split: str) -> list:
    """
    data 폴더에 JSON 파일이 없으면 Hugging Face에서 다운로드합니다.

    Args:
        split: "train", "valid", "Test"
    """
    if not isinstance(split, str) or not split:
        raise ValueError("split은 비어 있지 않은 문자열이어야 합니다.")

    filename = f"{split}.json"
    file_path = DATA_DIR / filename    

    if not file_path.is_file():
        print("json 파일이 존재하지 않습니다. 다운로드를 시작합니다.")

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(DATA_DIR)
            )
        except EntryNotFoundError as e:
            raise FileNotFoundError(
                f"Hugging Face 저장소에 '{filename}' 파일이 없습니다."
            ) from e
        
    with file_path.open("r", encoding=config.encoding_type) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"'{filename}'은 JSON 배열 형태여야 합니다.")

    return data

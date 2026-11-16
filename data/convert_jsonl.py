
import json
import ijson

from pathlib import Path
from huggingface_hub import hf_hub_download


REPO_ID = "jsuhong24/Learning-Crew-Dataset"


def convert_json_to_jsonl(split: str):
    #기존 Hugging Face 데이터 다운로드
    source_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=f"{split}.json",
        repo_type="dataset"
    )

    #JSONL 저장 경로
    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / f"{split}.jsonl"

    count = 0

    #기존 문서의 내용과 순서 유지
    with open(source_path, "rb") as source:
        with output_path.open("w", encoding="utf-8") as target:
            for document in ijson.items(source, "item", use_float=True):
                target.write(
                     json.dumps(document, ensure_ascii=False, default=float) + "\n"
                )
                count += 1

    print(f"{split}: {count}개 문서 변환 완료")
    print(f"저장 경로: {output_path}")


if __name__ == "__main__":
    convert_json_to_jsonl("train")
    convert_json_to_jsonl("valid")
    convert_json_to_jsonl("test")

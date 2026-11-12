#라이브러리 로딩
from pathlib import Path
from tempfile import NamedTemporaryFile     
import torch

#체크포인트 저장
def save_checkpoint(path: Path, state: dict) -> None:
    #임시파일, 학습 상태 저장 
    path.parent.mkdir(parents=True, exist_ok=True)  #저장할 폴더 생성
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:  #임시 파일 생성
        temporary_path = Path(file.name) #파일 이름으로 Path 객체로 변환

    try:
        torch.save(state, temporary_path)  #학습 상태 저장
        temporary_path.replace(path)  #임시 파일 -> 실제 파일로 변환
    finally:
        temporary_path.unlink(missing_ok=True) #임시 파일 삭제 


def load_checkpoint(path: str | Path, run_metadata: dict) -> dict:
    #path: 파일 경로, run_metadata: 모델 설정, 학습 설정 등 메타데이터
    state = torch.load(path, map_location="cpu", weights_only=True)
    mismatches = []
    for key, expected in run_metadata.items():  #key: _id 같은 항목 이름, expected 해당 항목 현재 값
        #예시
        #expected = {"seed": 42, "batch_size": 8}
        #saved = {"seed": 123, "batch_size": 8} -> 체크포인트에 저장된 설정
        saved = state.get(key)  #저장된 값 가져오기
        if isinstance(expected, dict) and isinstance(saved, dict):
            mismatches.extend(
                f"{key}.{field}"
                for field in sorted(expected.keys() | saved.keys())
                if field not in expected or field not in saved or expected[field] != saved[field]
                #field 또는 saved에 키가 없거나 두 값이 다르면 -> 학습 설정이 달라졌으면
            )
        elif expected != saved:
            mismatches.append(key)  #키 값부터 달랐을 때

    if mismatches:
    #리스트에 일치하지 않는 것이 하나라도 있을 때
        raise ValueError(f"Checkpoint mismatch: {', '.join(mismatches)}")       #
    return state
# 한국어 Tool Calling LoRA/QLoRA 실험 (RTX 3060 12GB)

로컬 12GB GPU에서 LoRA와 QLoRA를 직접 돌려 보고, 한국어 function calling에서 무엇이 좋아지고 무엇이 약해지는지 측정한 개인 프로젝트입니다.

## 요약

- 도구 선택과 JSON 출력 형식은 새로운 한국어 평가셋에서도 크게 좋아졌습니다. 1.5B LoRA는 도구명 일치율이 49.4%에서 92.4%로 올랐습니다.
- 한국어 인자 값 추출은 오히려 약해졌습니다. 7B에서 QLoRA의 인자 정확도(정답만 인정)는 형식 예시를 준 베이스보다 10.2%p 낮았습니다.
- 학습 정답 2,362건에 한글이 하나도 없었던 점과 함께 나타난 현상입니다. 인과는 확인하지 못했습니다.
- 시스템 프롬프트에 한국어 값을 번역하지 말라는 문장을 넣어도 거의 회복되지 않았습니다.

## 환경

- RTX 3060 12GB, RAM 16GB, WSL2 Ubuntu, Python 3.12
- torch 2.15, transformers 5.21, peft 0.21, trl 1.16, bitsandbytes 0.50
- 1.5B: Qwen2.5-1.5B-Instruct, LoRA(bf16)
- 7B: Qwen2.5-7B-Instruct, QLoRA(NF4, double quantization, compute bf16)

## 데이터

- 학습/평가 원본: Salesforce/xlam-function-calling-60k
- 영어 질의만 한국어로 번역하고 도구 정의는 영어로 유지했습니다. 번역은 로컬 Qwen2.5-7B-Instruct(4bit)로 했고, 중국어 토큰은 생성 단계에서 차단했습니다.
- 정답 인자 값이 질의에 그대로 들어 있는 샘플만 사용했습니다. 필터 후 학습 2,362건, 평가 241건입니다.
- 번역은 사람이 검수하지 않았습니다. 표본 20건씩 직접 읽었을 때 의미 오류가 약 10~15%였습니다.
- 외부 평가: Kakao FunctionChat-Bench SingleCall 500건(함수 25개 x 질의 4개 x 도구 목록 5종)

## 학습 설정

- LoRA r=16, alpha=32, dropout 0.05, 대상 all-linear
- 2 epoch, 유효 배치 16, lr 2e-4(cosine), warmup 10 step, max length 2048, gradient checkpointing
- 1.5B LoRA: 약 33~35분, VRAM 최대 5.12GB
- 7B QLoRA: 1시간 35분, VRAM 최대 7.19GB
- 평가는 greedy 디코딩(max_new_tokens 300)입니다.

## 결과

### xLAM 평가셋 (n=241, 완전 일치, 숫자 표기 정규화 후)

| 모델 | 전체 | 학습에 나온 도구 (n=188) | 새 도구 (n=53) |
|---|---|---|---|
| 1.5B 베이스 (형식 지시 없음) | 56.0% | 60.6% | 39.6% |
| 1.5B 베이스 + 형식 예시 | 70.5% | 72.3% | 64.2% |
| 1.5B LoRA | 85.1% | 87.8% | 75.5% |
| 1.5B LoRA 재실행 (같은 설정) | 85.9% | 89.4% | 73.6% |
| 7B 베이스 + 형식 예시 | 80.1% | 78.7% | 84.9% |
| 7B QLoRA | 92.5% | 93.1% | 90.6% |

새 도구 열은 표본이 작고, 방해용 도구로 학습 프롬프트에 나왔을 수 있습니다. 1.5B에서 베이스(형식 예시 포함) 대비 구조 오류(호출 개수 불일치, 질의에 없는 인자 추가, JSON 파싱 실패)는 46건에서 5건으로 줄었습니다.

### FunctionChat-Bench SingleCall (n=500, 고유 질의 100개)

| 모델 | 도구명 | 인자 (정답만) | 인자 (허용값 포함) | 한글 값이 출력에서 사라진 비율 |
|---|---|---|---|---|
| 1.5B 베이스 + 형식 예시 | 49.4% | 30.4% | 34.4% | 29.5% |
| 1.5B LoRA | 92.4% | 47.8% | 60.6% | 43.6% |
| 1.5B LoRA 재실행 | 93.0% | 49.4% | 60.6% | 40.5% |
| 7B 베이스 + 형식 예시 | 94.6% | 68.6% | 77.6% | 19.5% |
| 7B QLoRA | 98.8% | 58.4% | 73.4% | 49.8% |

한국어 값을 번역하지 말라는 문장을 시스템 프롬프트에 추가했을 때 7B 베이스는 18.3%, QLoRA는 46.7%로 거의 변하지 않았습니다.

질의 단위 부트스트랩 95% 구간으로 본 차이입니다.

- 1.5B LoRA 대 베이스, 도구명: +43.0%p [+36.8, +49.2]
- 7B QLoRA 대 베이스, 도구명: +4.2%p [+2.0, +7.0]
- 7B QLoRA 대 베이스, 인자(정답만): -10.2%p [-17.2, -3.4]
- 7B QLoRA 대 베이스, 인자(허용값 포함): -4.2%p [-10.6, +1.8] (0을 포함하므로 하락이라고 단정할 수 없습니다)

같은 설정으로 다시 학습한 결과(재실행)에서 한글이 사라진 비율이 3.1%p 달랐습니다. 이 정도 차이는 학습 잡음으로 보는 것이 맞습니다.

## 한계

- 학습 시드가 1개입니다. 부트스트랩 구간에는 학습 잡음이 들어 있지 않습니다.
- xLAM 평가셋은 학습 데이터와 같은 분포입니다.
- FunctionChat 500건은 고유 질의 약 100개를 도구 목록 5종으로 반복한 것이라 실질 표본이 약 100입니다.
- 번역을 검수하지 않았습니다.
- 채점은 직접 구현한 문자열 일치(대소문자 무시, 허용값 포함)입니다. 공식 평가는 GPT-4 심판이라 공식 점수와 비교할 수 없습니다.
- 도구 스키마를 xLAM 형식으로 변환하면서 required 정보가 default null로 표현되는 손실이 있습니다.
- 질의로는 알 수 없는 값(택시 요금 좌표, 일본 country_id 등)이 정답인 문항은 맞힐 수 없어서 점수 상한이 100%보다 낮습니다.
- 모델 크기와 양자화가 함께 바뀐 비교라서 QLoRA 자체의 효과로 읽으면 안 됩니다.

## 프로젝트 구조

```
src/
  common.py          # 프롬프트 구성, 응답 파싱, 모델 로딩 (학습/평가 공용)
  train.py           # LoRA/QLoRA 학습 (TRL SFTTrainer)
  eval_model.py      # xLAM 평가셋 평가
  eval_fcb.py        # FunctionChat-Bench 평가 (생성만 수행)
  score_fcb.py       # FunctionChat 채점
  diag_fcb.py        # 부트스트랩 신뢰구간, 한글 값 영어 전환 비율
  rescore.py         # xLAM 평가 재채점, 실패 유형 분류
  pipeline/          # 데이터 준비 (select_data, translate, filter_data, prepare_data, build_mix)
scripts/             # 데이터 구조 확인용 일회성 스크립트
data/                # 생성 데이터 (git 제외)
outputs/             # 어댑터, 평가 결과 (git 제외)
```

## 환경 구성 (Windows 11 + WSL2 기준)

### 1. WSL2와 메모리 설정

Windows PowerShell에서 Ubuntu를 설치합니다. NVIDIA 드라이버는 Windows에만 설치하고, WSL 안에는 설치하지 않습니다.

```powershell
wsl --install -d Ubuntu
```

RAM이 16GB라서 WSL 메모리를 제한합니다. `C:\Users\<사용자>\.wslconfig`를 만들고 아래 내용을 저장한 뒤 `wsl --shutdown`으로 적용합니다.

```ini
[wsl2]
memory=12GB
swap=8GB
```

Ubuntu 터미널에서 `nvidia-smi`에 GPU가 보이는지 확인합니다. 프로젝트는 `/mnt/c`가 아니라 WSL 홈 안에 두어야 파일 입출력이 느려지지 않습니다.

### 2. 패키지와 가상환경

```bash
sudo apt update && sudo apt install -y build-essential git curl tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/hwan6615/Korean-ToolCalling-LoRA.git ~/lora-project
cd ~/lora-project
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- 의존성은 Linux x86_64(WSL2 포함) 기준으로 고정했고 Python은 3.12를 사용했습니다. torch 2.15.0은 CUDA 13 계열 빌드가 설치되므로 CUDA 13을 지원하는 NVIDIA 드라이버가 필요합니다. nvidia-smi에서 CUDA 13.3이 표시되는 드라이버에서 확인했습니다.

### 3. 데이터 준비

xLAM 데이터셋은 접근 동의가 필요하므로 Hugging Face 계정으로 로그인하고 데이터셋 페이지에서 약관에 동의합니다.

```bash
hf auth login
mkdir -p data/external outputs/eval
git clone --depth 1 https://github.com/kakao/FunctionChat-Bench.git data/external/FunctionChat-Bench
```

모든 명령은 프로젝트 루트에서 `python -m`으로 실행합니다.

## 실행 방법

### 데이터 생성

```bash
python -m src.pipeline.select_data      # xLAM에서 샘플 선별
python -m src.pipeline.translate        # 질의 번역 (이어서 실행 가능)
python -m src.pipeline.filter_data      # 번역 검증/필터, clean.jsonl 생성
python -m src.pipeline.prepare_data     # 학습용 chat 포맷 변환
```

번역은 스크립트 안에 고정된 Qwen2.5-7B-Instruct(4bit)를 쓰며, 아래 환경변수의 영향을 받지 않습니다. 3,300건 번역에 약 3.6시간이 걸렸습니다.

### 학습

환경변수로 모델과 설정을 바꿉니다.

| 변수 | 의미 | 기본값 |
|---|---|---|
| MODEL_ID | 베이스 모델 | Qwen/Qwen2.5-1.5B-Instruct |
| USE_4BIT | 1이면 4bit로 불러옴 (QLoRA) | 끔 |
| RUN_NAME | 결과 폴더 이름 (outputs/RUN_NAME) | qwen1.5b-lora |
| BATCH, ACCUM | 장치당 배치, 누적 스텝 | 4, 4 |
| EPOCHS | 에폭 수 | 2 |
| TRAIN_FILE | 학습 데이터 경로 | data/train_sft.jsonl |

```bash
# 1.5B LoRA (bf16)
python -m src.train 2>&1 | tee outputs/train_1.5b.log

# 7B QLoRA (4bit), 유효 배치를 같게 맞춤
MODEL_ID=Qwen/Qwen2.5-7B-Instruct USE_4BIT=1 RUN_NAME=qwen7b-qlora BATCH=2 ACCUM=8 \
  python -m src.train 2>&1 | tee outputs/train_7b.log
```

장시간 실행은 tmux 안에서 돌리면 터미널 연결이 끊겨도 유지됩니다. 학습 시작 직후 `학습 데이터: ..., N건`이 출력되는지 확인합니다.

### 평가

```bash
# xLAM 평가셋
python -m src.eval_model --adapter outputs/qwen1.5b-lora/final --name lora_final
# FunctionChat-Bench (생성 후 채점)
python -m src.eval_fcb --adapter outputs/qwen1.5b-lora/final --name lora_final
python -m src.diag_fcb      # 신뢰구간과 한글 값 영어 전환 비율
python -m src.rescore       # xLAM 재채점과 실패 유형
```

베이스 모델 평가는 `--adapter` 없이 `--hint`(출력 형식 예시를 시스템 프롬프트에 추가)를 붙입니다. 한국어 값을 번역하지 말라는 문장을 추가하려면 `--note`를 붙입니다. 7B 평가는 평가 명령 앞에 `MODEL_ID=Qwen/Qwen2.5-7B-Instruct USE_4BIT=1`을 붙입니다. `diag_fcb.py`, `score_fcb.py`, `rescore.py`의 `RUNS` 목록에 모델 이름을 추가해야 표에 나옵니다.

### 소요 시간 (RTX 3060 12GB에서 측정)

- 1.5B LoRA 학습: 약 33~35분, VRAM 최대 5.12GB
- 7B QLoRA 학습: 1시간 35분, VRAM 최대 7.19GB
- FunctionChat 500건 생성: 1.5B 약 8~11분, 7B 약 10~18분

## 다음 계획

- 한국어 값이 들어 있는 공개 데이터(palette-lab/songgot-tools-ko)를 일부 섞어 재학습하고 같은 평가로 비교합니다. 데이터 생성 스크립트(`src/pipeline/build_mix.py`)는 완성했고, 학습은 아직 하지 않았습니다.
- 학습 데이터 양을 바꿔 가며 곡선을 그립니다.
- 호출하지 않아야 하는 경우와 한영 매핑이 필요한 값을 평가에 추가합니다.

## 출처와 라이선스

- Salesforce/xlam-function-calling-60k (CC-BY-4.0, APIGen 논문 인용)
- Kakao FunctionChat-Bench (Apache-2.0)
- palette-lab/songgot-tools-ko (Apache-2.0, 현재 결과에는 사용하지 않음)
- Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct (Apache-2.0)
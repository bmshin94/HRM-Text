# HRM-Text 전수조사 분석 & 활용/수익화 정리

> 작성: Claude Code (카리나 페르소나) · 요청자: mono7594@gmail.com
> 작성일: 2026-09-27
> 대상 저장소: **https://github.com/bmshin94/HRM-Text**
> 원본(Upstream): **https://github.com/sapientinc/HRM-Text**
> 작업 브랜치: `claude/serene-hopper-7j823d`

---

## 목차

1. [프로젝트 정체 및 개요](#1-프로젝트-정체-및-개요)
2. [핵심 아이디어: HRM 아키텍처](#2-핵심-아이디어-hrm-아키텍처)
3. [저장소 전수조사 (폴더/파일별)](#3-저장소-전수조사-폴더파일별)
4. [쉬운 설명 (비유 버전)](#4-쉬운-설명-비유-버전)
5. [설치 및 사용법](#5-설치-및-사용법)
6. [플러그인/스킬/MCP 여부](#6-플러그인스킬mcp-여부)
7. [API 토큰 필요 여부](#7-api-토큰-필요-여부)
8. [GitHub에서 유명한 이유](#8-github에서-유명한-이유)
9. [로컬 에이전트 구축 적합성](#9-로컬-에이전트-구축-적합성)
10. [React / PHP 로 구현 가능한가](#10-react--php-로-구현-가능한가)
11. [수익화 아이디어 상세](#11-수익화-아이디어-상세)
12. [리스크 및 주의사항](#12-리스크-및-주의사항)
13. [참고 링크](#13-참고-링크)

---

## 1. 프로젝트 정체 및 개요

| 항목 | 내용 |
| --- | --- |
| 정식 명칭 | **HRM-Text: Efficient Pretraining Beyond Scaling** |
| 개발 주체 | Sapient Intelligence (`sapientinc`) |
| 논문 | arXiv `2605.20613` |
| 공개 모델 | HuggingFace `sapientinc/HRM-Text-1B` |
| 라이선스 | **Apache License 2.0** (상업적 이용 가능) |
| 구성 언어 | Python 100% (파일 53개 / Python 코드 약 3,048줄) |
| 성격 | 앱/툴이 아닌 **LLM 사전학습(pretraining) 프레임워크** |

### 한 줄 요약

> 파라미터 10억(1B) 규모 언어모델을 H100 8~16장으로 약 2일, **기존 대비 컴퓨팅 130~600배 / 데이터 150~900배 적게** 사용해 처음부터(from scratch) 학습하는 전체 파이프라인.

### 공개 벤치마크 (README 기준, 저자 자체 측정)

| Size | GPUs | Time | 비용(추정) | GSM8k | MATH | DROP | MMLU | ARC-C | HellaSwag | Winogrande | BoolQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L (0.6B) | 8 | 50h | ~$800 | 77.6% | 51.2% | 78.6% | 56.6% | 75.9% | 52.7% | 67.6% | 85.0% |
| XL (1B) | 16 | 46h | ~$1,472 | 84.7% | 56.5% | 82.3% | 60.7% | 81.9% | 63.4% | 72.4% | 86.2% |

*H100 시간당 $2 기준 추정.*

### 언제 쓰는가

**적합한 경우**
- 도메인 전용 언어모델을 처음부터 만들고 싶을 때 (법률/의료/제조/한국어 등)
- 파운데이션 모델 연구, 논문 작성
- 공개된 HRM-Text-1B를 자체 데이터로 SFT(지도 파인튜닝)하여 특화
- "작은 모델로 추론 성능을 끌어올리는 방법" 연구

**부적합한 경우**
- 단순 챗봇/앱 개발 (상용 API 또는 Ollama가 적합)
- Hopper 세대 GPU가 없는 환경 (FlashAttention 3 의존)
- RAG / 에이전트 애플리케이션 개발 자체

---

## 2. 핵심 아이디어: HRM 아키텍처

일반 Transformer는 층을 한 번 통과해 출력한다. HRM은 **같은 층 묶음을 여러 번 순환(recurrent)** 시켜 "잠재 공간에서의 사고 반복"을 구현한다.

`models/baselines/hrm_nocarry_bp_warmup.py` (100줄) 기준 구조:

```
H_level (상위/느린 사고)  : n_layers / 2 개 층  → 큰 방향 설정
L_level (하위/빠른 사고)  : n_layers / 2 개 층  → 세부 계산

for i in range(H_cycles):          # 기본 2
    for k in range(L_cycles):      # 기본 3
        z_L = L_level(z_L, z_H)    # 세부 갱신
    z_H = H_level(z_H, z_L)        # 큰 그림 갱신
return z_H
```

### 효과

- XL(32층) 설정 + `half_layers: true` → H 16층 / L 16층
- 실제 블록 통과 횟수 = L 6회 + H 2회 = **8회** → 유효 추론 깊이 약 **128 layer-applications**
- 즉 **파라미터 수는 유지하면서 추론 깊이만 확장** → 데이터·컴퓨팅 효율의 근원

### 주요 구현 디테일

| 기법 | 파일 | 설명 |
| --- | --- | --- |
| **BP warmup (절단 BPTT)** | `hrm_nocarry_bp_warmup.py` | 학습 초반 `bp_min_steps=2`부터 `bp_max_steps=5`까지 역전파 구간을 점진 확대(`bp_warmup_ratio=0.2`). 메모리 절감 + 안정화 |
| **no-carry** | 동일 | 원본 HRM의 carry 상태를 제거(`initial_carry -> None`)해 구조 단순화 |
| **zL_init 버퍼** | 동일 | L 레벨 초기 은닉 상태를 학습 가능한 버퍼로 보유 |
| **PrefixLM 2-pass attention** | `models/flash_attention_prefixlm_v2.py` | 프롬프트 영역은 양방향, 응답 영역은 인과(causal) 어텐션 |
| **Multipack (LPT) 배치** | `multipack_sampler.py` | 토큰 슬롯 활용률 향상 + 2차 어텐션 연산 균형 분배 |
| **FSDP2 + torch.compile** | `pretrain.py` | 분산 샤딩 학습, `reshard_after_forward=False`로 VRAM↔통신 트레이드오프 |
| **AdamATan2 + EMA** | `models/adam_atan2.py` | 옵티마이저 내부에 EMA 버퍼 내장(`ema: 0.9999`), 평가/배포 시 EMA 가중치 기본 사용 |
| **RoPE / SwiGLU / gated MHA** | `models/layers.py` | 현대 LLM 표준 구성요소 + static KV cache |

---

## 3. 저장소 전수조사 (폴더/파일별)

```text
HRM-Text/
├── pretrain.py                        # 학습 진입점: FSDP2, LR 스케줄, W&B, DCP 체크포인트
├── dataset_new.py                     # PrefixLM 패킹 데이터셋 (tokens.npy + epoch 인덱스)
├── multipack_sampler.py               # 분산 multipack 배치 샘플러 (LPT 할당)
├── simple_inference_engine.py         # 체크포인트 로드 + compiled 생성 엔진 (KV캐시/Gumbel 샘플링)
├── requirements.txt                   # torch, flash_attn_3, vllm, wandb, hydra, lm-eval 등 19개
│
├── models/
│   ├── baselines/
│   │   ├── hrm_nocarry_bp_warmup.py   # ★ HRM-Text 본체 (100줄)
│   │   ├── trm_nocarry.py             # Tiny Recursive Model 베이스라인
│   │   ├── ut_nocarry.py              # Universal Transformer 베이스라인
│   │   ├── rins_nocarry.py            # Recursive Inference Scaling 베이스라인
│   │   └── transformer_wrapper.py     # 표준 Transformer 베이스라인
│   ├── transformer.py                 # TransformerBlock / Cache / TransformerConfig
│   ├── layers.py                      # RoPE, Attention, SwiGLU, KV cache, init 유틸
│   ├── flash_attention_prefixlm_v2.py # PrefixLM 2-pass FA3 경로
│   ├── lm_head.py                     # 스케일 임베딩 + 출력 헤드 + CE loss + accuracy
│   ├── adam_atan2.py                  # EMA 내장 옵티마이저
│   └── common.py                       # IGNORE_LABEL_ID, packing 유틸
│
├── config/                            # Hydra 설정
│   ├── cfg_pretrain.yaml              # lr 2.2e-4, epochs 4, global_batch_size 196608, ema 0.9999
│   ├── cfg_sft.yaml                   # lr 3e-5, epochs 5, batch 32768, ema 0.999, bp_warmup 0
│   ├── arch/net/{hrm,transformer,trm,trm_match_recurrence,rins,ut}.yaml
│   ├── arch/size/{B,L,XL,XXL,XXL_wide}.yaml
│   └── data/{hlm,sft}.yaml            # hlm: /dev/shm/sampled (인메모리)
│
├── evaluation/
│   ├── main.py                        # 평가 진입점 (ckpt_epoch 미지정 시 최신 자동 선택)
│   ├── benchmarks.py                  # GSM8k, MATH, DROP, MMLU, MMLUPro, ARC, HellaSwag,
│   │                                  #   Winogrande, BoolQ, AIMEMajorityVoting
│   ├── engines.py                      # SimpleEngine(자체) / vLLM 엔진
│   ├── config/hrm_benchmarking.yaml   # condition "synth,cot" / "direct", max_context 3072~4096
│   ├── config/hrm_maj_vote_benchmarking.yaml
│   └── config/vllm_benchmarking.yaml  # Llama3.2-3B, Qwen3.5-2B, Olmo-3-7B, Ouro-1.4B 비교
│
├── scripts/
│   ├── prepare_sft_data.py            # ★ JSONL → V1Dataset 바이너리 변환
│   └── test_prepare_sft_filter.py     # 빈 response 필터 기능 테스트
│
├── conversion/convert_to_hf.py        # FSDP2 체크포인트 → HF safetensors 포맷 내보내기
├── docker/Dockerfile                  # python3.13 + CUDA 12.8.1 + PyTorch cu128 + FA3
├── docker/requirements/torch_extensions.txt
├── .github/workflows/docker_build_push.yml  # main 브랜치 push 시 도커 이미지 빌드/푸시
├── assets/{banner.png, benchmark_scatter.png}
├── CLAUDE.md                          # 사용자가 추가한 카리나 페르소나 설정 (프로젝트 본체와 무관)
├── LICENSE                            # Apache 2.0
└── README.md
```

### 모델 사이즈 프리셋

| Config | Layers | Hidden | Heads |
| --- | ---: | ---: | ---: |
| B | 12 | 1024 | 8 |
| L | 24 | 1280 | 10 |
| XL | 32 | 1536 | 12 |
| XXL | 72 | 1792 | 14 |
| XXL_wide | 32 | 2560 | 20 |

*HRM / RINS는 `half_layers: true`로 위 층수를 H·L에 절반씩 분배.*

### 사용자 포크의 커밋 히스토리 (참고)

```
4a5abef Merge pull request #1 from bmshin94/feat/claude-guide
2cea2dc docs: created CLAUDE.md persona guide
aaa948e Update HRM-Text banner
5802461 Add experimental branch in README.md
da566c9 (upstream) Merge PR #11 filter-empty-response-samples ...
```

---

## 4. 쉬운 설명 (비유 버전)

### "AI 학원" 비유

이 저장소는 AI 자체가 아니라 **AI를 가르치는 학원 시스템 전체**다.

| 폴더/파일 | 비유 |
| --- | --- |
| `models/` | 학생의 뇌 구조 설계도 |
| `dataset_new.py` | 교재를 학생이 먹기 좋게 잘라주는 사람 |
| `pretrain.py` | 수업을 진행하는 선생님 (조교 8~16명과 함께) |
| `config/` | 시간표와 커리큘럼 |
| `evaluation/` | 모의고사 채점관 |
| `conversion/` | 졸업장 발급처 (남들이 알아보는 포맷) |
| `docker/` | 학원 건물 (환경 세팅) |

### HRM이 똑똑한 이유 — "시험 문제 푸는 방식" 비유

- **일반 GPT**: 문제를 보고 머릿속에서 한 번 쭉 처리 후 바로 답을 쓴다. ("직관형")
- **HRM**: "이건 방정식 문제구나"(H) → "x 옮기고, 2로 나누고"(L×3) → "다음 단계로"(H) → … → 답. ("연습장에 끄적이는 형")

뇌 크기(파라미터)는 같지만 머릿속에서 여러 번 굴리므로 추론 문제에 강하다. 그래서 1B 모델이 GSM8k 84%를 기록한다.

### 비용 감각

```
GPT-4급 모델 학습        : 수백억 ~ 수천억원
일반적인 1B 모델 학습    : 수조 토큰 + GPU 수백 장
HRM-Text 1B 학습        : 약 $1,000 ~ $1,500 + H100 16장 × 2일
```

핵심 임팩트: **대기업 전유물이었던 사전학습을 개인/스타트업 영역으로 끌어내렸다.**

---

## 5. 설치 및 사용법

### 방법 A. 완성 모델만 사용 (가장 가벼움)

```bash
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("sapientinc/HRM-Text-1B", trust_remote_code=True)
tok   = AutoTokenizer.from_pretrained("sapientinc/HRM-Text-1B")
```

> README `Status` 기준: **Native Transformers 지원은 다음 릴리스에 포함 예정**, **vLLM 지원은 진행 중**.
> 따라서 현재는 `trust_remote_code` 또는 본 저장소의 `simple_inference_engine.py` 경유가 필요할 수 있다.

### 방법 B. Docker (공식 권장)

```bash
git clone https://github.com/bmshin94/HRM-Text
cd HRM-Text

docker run --gpus all --ipc=host --network=host -it \
  -v "$PWD":/workspace \
  sapientai/hrm-text:latest
```

멀티노드는 모든 노드에 동일 경로로 공유 스토리지를 마운트하는 것을 권장:

```text
/shared/
├── HRM-Text/
│   └── checkpoints/
└── data_io/
```

### 방법 C. 소스 설치

```bash
# 1) PyTorch + CUDA 12.8 + FlashAttention 3 (버전은 docker/Dockerfile 참고)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# 2) 나머지 의존성
pip install -r requirements.txt
```

### 전체 파이프라인

```bash
# [1] 데이터 준비 (별도 저장소: sapientinc/data_io)
cd <DATA_IO_PATH>
python sample_tokenized.py epochs=4 output_path=/dev/shm/sampled > show_analytics.md

# [2] W&B 로그인 (학습 필수)
wandb login <API_KEY>        # 키 발급: https://wandb.ai/authorize

# [3] 사전학습 — L 사이즈 / H100 8장 단일 노드
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 pretrain.py \
  arch/size@arch=L lr=2.5e-4 global_batch_size=172032

# [3-b] XL 사이즈 / H100 8장 × 2노드 (각 노드에서 실행)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=<RANK> \
  --master_addr=<ADDR> --master_port=<PORT> pretrain.py

# [4] 평가 (GPU 1장 / 80GB 기준)
python -m evaluation.main ckpt_path="checkpoints/..."
python -m evaluation.main ckpt_path="..." run_only=[MATH,DROP,ARC,MMLU]
# OOM 시: generation_config.batch_size=16

# [5] HuggingFace 포맷 내보내기
python -m conversion.convert_to_hf --ckpt_path "checkpoints/..." --out_dir "<OUT>"
```

### 파인튜닝(SFT) — 실무에서 가장 현실적인 경로

입력 JSONL 포맷 (한 줄에 한 객체):

```json
{"instruction": "<full prompt>", "response": "<expected output>", "condition": "direct"}
```

```bash
# 1) 데이터 변환
python scripts/prepare_sft_data.py \
  --train my_data.jsonl \
  --tokenizer /path/to/tokenizer.json \
  --output /dev/shm/sft_data \
  --epochs 5

# 2) 학습
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
torchrun --nproc_per_node=8 pretrain.py \
  --config-name cfg_sft \
  arch/size@arch=XL \
  data.path=/dev/shm/sft_data \
  resume_from=/path/to/pretrain_ckpt \
  weights_only_resume_from_ema=true \
  +checkpoint_path=/path/to/sft_out
```

### 자주 밟는 함정

1. `prepare_sft_data.py --epochs` 값은 학습 config의 `epochs`와 **반드시 일치**해야 한다(에폭별 사전 셔플을 미리 생성).
2. `arch/*` 오버라이드는 체크포인트의 `all_config.yaml`과 정확히 일치해야 한다(`arch.n_layers` 등 개별 오버라이드 필요할 수 있음).
3. `/dev/shm`은 램디스크이므로 **충분한 RAM**이 필요하다.
4. `resume_from`은 옵티마이저 상태(EMA 포함)까지 로드한다. 사전학습 EMA에서 새로 시작하려면 `weights_only_resume_from_ema=true`.
5. 멀티노드에서 각 노드는 자기 샤드만 저장하므로 **공유 스토리지 마운트 권장**.
6. `--context-size`는 최대 샘플 길이 + 1 이상이어야 한다(autoregressive shift).

---

## 6. 플러그인/스킬/MCP 여부

| 분류 | 해당 여부 | 근거 |
| --- | --- | --- |
| Claude Code 플러그인 | ❌ | `.claude-plugin/`, `plugin.json` 없음 |
| Claude Skill | ❌ | `SKILL.md` 없음 |
| MCP 서버 | ❌ | MCP 서버 구현/설정 없음 |
| **ML 학습 프레임워크** | ✅ | PyTorch + Hydra 기반 사전학습/평가 코드 |

혼동의 원인은 사용자가 직접 추가한 `CLAUDE.md`(페르소나 설정 파일, 커밋 `2cea2dc`)이며, 이는 프로젝트 본체와 무관하다.

> 확장 아이디어: `simple_inference_engine.py`를 MCP 서버로 래핑하면 Claude Code / Cursor에서 로컬 HRM 모델을 도구로 호출할 수 있다.

---

## 7. API 토큰 필요 여부

| 토큰 | 필요도 | 용도 | 없을 때 |
| --- | --- | --- | --- |
| **W&B API Key** | 학습 시 필수 | `pretrain.py`의 `wandb.init()` | 학습 시작 불가 (`WANDB_MODE=offline`로 우회 가능) |
| HuggingFace Token | 조건부 | 게이트 데이터셋/모델 다운로드 | 공개 데이터만 사용 시 불필요 |
| Docker Hub Token | CI 전용 | `.github/workflows/docker_build_push.yml` 시크릿 | 포크에서는 무관 |
| Claude / OpenAI Key | **불필요** | — | 모델을 직접 실행하므로 외부 LLM API 불필요 |

---

## 8. GitHub에서 유명한 이유

1. **"$1,000로 파운데이션 모델"이라는 충격적 헤드라인** — 사전학습의 심리적/금전적 장벽을 무너뜨림.
2. **HRM 계보의 후광** — 원본 HRM(2025)이 27M 파라미터로 ARC-AGI 퍼즐을 풀어 큰 화제. "스케일링이 유일한 답이 아닐 수 있다"는 메시지.
3. **스케일링 법칙에 대한 반론 서사** — `Efficient Pretraining Beyond Scaling`이라는 제목 자체가 선언문.
4. **완전 공개** — 논문 + 학습 코드 + 학습된 가중치(HF) + Docker 이미지 + 평가 코드 + Apache 2.0. 재현 가능한 전체 레시피.
5. **코드가 매우 간결** — HRM 본체 100줄, 전체 약 3,000줄. 진입장벽이 낮음.
6. **베이스라인 5종 동봉** — transformer / trm / trm_match_recurrence / rins / ut 를 동일 코드베이스에서 공정 비교.
7. **커뮤니티 운영** — Discord 1,200명+, 커뮤니티 기여 아키텍처를 `experimental/*` 브랜치로 수용(예: `experimental/moe-64x8`, MoE 64×8).

> 유의: 성능 향상이 "계층적 추론 구조" 때문인지 "순환/파라미터 재사용 + 잘 튜닝된 학습 레시피" 때문인지에 대해서는 학계 논쟁이 있다. 벤치마크 수치는 저자 자체 측정이므로 사업 적용 전 독립 검증이 필요하다.

---

## 9. 로컬 에이전트 구축 적합성

### 유리한 점

| 항목 | 내용 |
| --- | --- |
| 모델 크기 | 1B → FP16 약 2GB. 소비자용 GPU에도 적재 가능 |
| 추론 성능 | 1B급에서 GSM8k 84%는 구조적 출력(JSON/도구 호출)에 유리할 가능성 |
| 비용 | API 호출 비용 0원 (에이전트는 호출 횟수가 많아 효과 큼) |
| 보안 | 완전 오프라인, 데이터 외부 유출 없음 → 사내/규제 환경에 적합 |
| 커스터마이즈 | SFT 파이프라인 완비 (JSONL → 학습) |

### 불리한 점

| 항목 | 내용 |
| --- | --- |
| vLLM 미지원 | README 기준 "진행 중". 현재는 자체 `simple_inference_engine.py` 사용 |
| Ollama / llama.cpp 미지원 | GGUF 양자화 경로 없음 → CPU/맥 추론 불가 |
| 실효 연산량 | H2×L3 = 블록 8회 통과 → 1B 파라미터지만 연산량은 훨씬 큼 (지연 시간 주의) |
| 컨텍스트 길이 | 평가 설정상 3,072~4,096 토큰. 에이전트 용도로는 짧음 |
| Tool calling | 별도 SFT 없이는 함수 호출 포맷 미학습 |
| 하드웨어 | FlashAttention 3 → Hopper 세대 GPU 전제 |

### 결론

- **지금 당장 범용 로컬 에이전트**가 목표라면 Ollama + Qwen3 / Llama 3.2 계열이 생태계 완성도에서 유리.
- HRM-Text는 **"한 도메인만 매우 잘하는 초경량 전문가 모델"** 로 활용하는 것이 합리적. 즉 에이전트의 범용 두뇌가 아니라, 에이전트 내부의 **전문 부품**(문서 분류기, SQL 생성기, 코드 매핑기, NPC 대화 등).

---

## 10. React / PHP 로 구현 가능한가

### 모델 학습/추론 자체 — 불가능

- FlashAttention 3은 C++/CUDA 커널이며 JS/PHP 바인딩이 없다.
- `torch.compile`, FSDP2, `torch.distributed.checkpoint` 등은 Python 전용이다.
- bf16 텐서 / KV 캐시 등 GPU 메모리 관리를 PHP에서 수행할 수 없다.

### 래핑(제품화) — 완전히 가능

```text
┌──────────────────────────────────────┐
│  React 프론트엔드                      │  채팅 UI, 스트리밍, 관리자 대시보드
└───────────────┬──────────────────────┘
                │ REST / SSE / WebSocket
┌───────────────▼──────────────────────┐
│  PHP (Laravel) 백엔드                 │  인증, 결제, 사용량 과금, 대화 로그 DB
└───────────────┬──────────────────────┘
                │ 내부 HTTP
┌───────────────▼──────────────────────┐
│  Python 추론 서버 (FastAPI)           │  ← Python이 필요한 유일한 레이어
│  └ simple_inference_engine.py         │
└───────────────┬──────────────────────┘
                │
             GPU 서버
```

Python 측 최소 구현:

```python
# inference_server.py
from fastapi import FastAPI
from simple_inference_engine import inference_load_checkpoint, inference_generate

app = FastAPI()
ckpt = inference_load_checkpoint("./checkpoints/xl", None, ckpt_use_ema=True)

@app.post("/generate")
def generate(prompt: str, condition: str = "direct"):
    gen = inference_generate(
        ckpt, iter([(0, (condition, prompt))]),
        max_tokens=3072, max_generation=1024, batch_size=1,
    )
    return {"text": next(gen)[1]}
```

PHP(Laravel) 측 프록시 + 과금:

```php
$res = Http::timeout(120)->post('http://gpu-server:8000/generate', [
    'prompt'    => $request->input('prompt'),
    'condition' => 'direct',
]);
Usage::increment($user->id);
return $res->json();
```

React 측 호출:

```jsx
const ask = async (prompt) => {
  const r = await fetch('/api/ask', { method: 'POST', body: JSON.stringify({ prompt }) });
  const { text } = await r.json();
  setMessages(m => [...m, { role: 'assistant', content: text }]);
};
```

> `inference_generate`는 제너레이터이므로 SSE 기반 **토큰 단위 스트리밍**을 붙이기 쉽다.
> 핵심 전략: **Python은 GPU 앞의 얇은 레이어 한 겹만**, 비즈니스 로직과 UI는 React + PHP로.

---

## 11. 수익화 아이디어 상세

### 1) 버티컬 SLM (도메인 특화 초경량 모델) 라이선스

| 항목 | 내용 |
| --- | --- |
| 초기 투자 | $500 ~ $1,500 (GPU 클라우드 + 데이터 정제) |
| 수익 모델 | 연 라이선스 500만~3,000만원/사, 또는 온프레미스 구축비 |
| 소요 기간 | 2~3개월 |
| 난이도 | ★★★ |

기업의 3대 통증(데이터 외부 유출 우려 / API 비용 / 응답 지연)을 "자사 서버 내 전용 모델"로 해소한다.

**타겟 도메인(국내)**: 의료(진료기록 요약, KCD 코드 매핑), 법률(계약 조항 분류·리스크 탐지), 제조(설비 매뉴얼 QA, 불량 리포트 분류), 금융(컴플라이언스 문서 검토), 이커머스(상품 카테고리 자동분류, 리뷰 감정분석).

### 2) "자사 전용 LLM 구축" SI / 컨설팅

| 항목 | 내용 |
| --- | --- |
| 초기 투자 | 거의 0원 (레퍼런스 1건만 확보) |
| 수익 모델 | 프로젝트당 3,000만~2억원 + 유지보수 월 200~500만원 |
| 소요 기간 | 즉시 시작 가능 |
| 난이도 | ★★ (영업력이 관건) |

세일즈 포인트: "GPT API 월 500만원 → 초기 구축 5,000만원 + 이후 월 전기료 수준, 데이터는 사외 반출 없음." ROI 계산이 명확해 설득이 쉽다.

### 3) SFT-as-a-Service (파인튜닝 자동화 SaaS)

| 항목 | 내용 |
| --- | --- |
| 초기 투자 | $2,000 ~ $5,000 (GPU + 개발) |
| 수익 모델 | 학습 1건당 30만~200만원 / 구독 월 50만원~ |
| 난이도 | ★★★★ |

```text
React 대시보드 (업로드 / 진행률 / 플레이그라운드)
  → PHP Laravel (회원·결제·작업 큐·사용량)
  → Python 워커 (prepare_sft_data.py → pretrain.py → convert_to_hf.py)
  → GPU (온디맨드 임대 → 고정비 최소화)
```

`scripts/prepare_sft_data.py` + `config/cfg_sft.yaml` 조합이 이미 자동화에 적합한 구조이므로, 프론트/백엔드 강점을 그대로 활용할 수 있는 선택지다.

### 4) 한국어 특화 HRM-Text

| 항목 | 내용 |
| --- | --- |
| 초기 투자 | $1,500 ~ $3,000 |
| 수익 모델 | 모델 라이선스 + 정부/기관 R&D 과제 + 인지도를 통한 컨설팅 유입 |
| 난이도 | ★★★★ |

"한국어 1B로 GSM8k 80%" 급 결과는 국내 AI 생태계에서 즉각적인 인지도를 만든다. 한국어 토크나이저 최적화만으로도 차별화 가능하며, 확보한 레퍼런스는 2)번 SI 수주에 직접 기여한다.

### 5) 엣지 / 온디바이스 임베딩 라이선스

| 항목 | 내용 |
| --- | --- |
| 초기 투자 | $1,000 + 양자화 작업 |
| 수익 모델 | 기기당 로열티 (대당 1,000~10,000원) — 물량 확보 시 수익성 최상 |
| 난이도 | ★★★★★ |

타겟: 키오스크, 로봇청소기, 차량 인포테인먼트, 산업용 HMI, 의료기기, 국방.
선행 과제: GGUF/ONNX 변환 + 양자화 파이프라인 구축(현재 미지원 → 직접 구현 시 그 자체가 차별화 및 오픈소스 기여).

### 6) 교육 콘텐츠 / 강의 — 현금화 최속

| 항목 | 내용 |
| --- | --- |
| 초기 투자 | 100만원 이하 |
| 수익 모델 | 온라인 강의 (예: 수강생 500명 × 10만원 = 5,000만원) |
| 소요 기간 | 1~2개월 |
| 난이도 | ★★ |

"파인튜닝" 강의는 포화 상태이나 **"사전학습 전 과정"** 강의는 희소하다.
커리큘럼: FSDP2 분산학습 / FlashAttention 3 / 시퀀스 패킹 / BP warmup / 벤치마크 평가 / HF 배포.

### 7) HuggingFace 허브 전략 (간접 수익)

파생 모델을 무료 공개하여 다운로드 수를 포트폴리오로 활용. 업스트림 기여(`experimental/*` 브랜치)는 신뢰도를 크게 높이고 해외 컨설팅 유입 경로가 된다.

### 8) MCP 서버 상품화

```text
Claude Code / Cursor ──MCP──> 로컬 HRM 추론 서버
                              (사내 문서 검색·분류, 오프라인, 과금 없음)
```

개발도구 시장은 지불 의사가 높다. 구독 월 2~5만원 × 1,000명 = 월 2,000~5,000만원 규모가 가능하며, React/PHP로 관리 UI를 붙여 완성도를 높일 수 있다.

### 우선순위 로드맵

```text
[1개월]     6) 교육 콘텐츠        → 현금 확보 + 전문성 증명
[2~3개월]   2) SI / 컨설팅        → 레퍼런스 1건 확보 (소형 SFT로 충분)
[4~6개월]   3) SFT SaaS           → React + PHP 역량 직접 활용
[6개월~]    1) 버티컬 SLM / 4) 한국어 HRM → 스케일업
```

---

## 12. 리스크 및 주의사항

| 리스크 | 대응 |
| --- | --- |
| 벤치마크 수치가 저자 자체 측정 | 사업화 전 독립 재현/검증. 성능 보증 계약 전 필수 |
| HRM 구조의 효과에 학계 논쟁 존재 | "HRM"을 마케팅 전면에 세우기보다 결과물 품질로 승부 |
| FlashAttention 3 → Hopper GPU 의존 | 학습은 클라우드 온디맨드, 추론은 양자화/커널 대체로 우회 |
| 생태계 미성숙 (vLLM / Ollama 미지원) | 초기에는 자체 서빙 코드 유지보수 비용 발생 |
| 컨텍스트 3~4K 토큰 | 장문 처리 업무는 RAG/청킹으로 보완 |
| 오픈소스라 진입장벽 낮음 | 차별화 요소는 코드가 아니라 **데이터 + 도메인 지식 + 제품화 실행력** |
| 데이터 준비에 별도 저장소 필요 | `sapientinc/data_io` 파이프라인 선행 구축 필요 |
| 학습 시 W&B 의존 | 폐쇄망이면 `WANDB_MODE=offline` 또는 로깅 코드 우회 필요 |

---

## 13. 참고 링크

| 항목 | URL |
| --- | --- |
| **본 저장소 (포크)** | https://github.com/bmshin94/HRM-Text |
| 원본 저장소 (Upstream) | https://github.com/sapientinc/HRM-Text |
| 데이터 파이프라인 | https://github.com/sapientinc/data_io |
| 논문 (arXiv) | https://arxiv.org/abs/2605.20613 |
| 공개 모델 (HuggingFace) | https://huggingface.co/sapientinc/HRM-Text-1B |
| 커뮤니티 MoE 체크포인트 | https://huggingface.co/Xiaoye08/HRM-MoE |
| 실험 브랜치 (MoE 64×8) | https://github.com/sapientinc/HRM-Text/tree/experimental/moe-64x8 |
| Discord 커뮤니티 | https://discord.gg/sapient |
| Weights & Biases | https://wandb.ai/authorize |

---

## 부록: 한눈에 보는 결론

- **정체**: LLM 사전학습 프레임워크 (플러그인/스킬/MCP 아님)
- **강점**: 저비용 사전학습, 1B급에서 뛰어난 추론 성능, Apache 2.0, 간결한 코드
- **약점**: Hopper GPU 의존, 짧은 컨텍스트, 서빙 생태계 미성숙
- **가장 현실적인 활용**: 공개 1B 체크포인트 + 자체 데이터 SFT → 도메인 특화 초경량 모델
- **가장 빠른 수익화**: 교육 콘텐츠 → SI 컨설팅 → SFT SaaS
- **기술 스택 전략**: GPU 앞단만 Python(FastAPI), 나머지는 React + PHP

# PROMPTPEEK Attack Implementation Guidelines (v1.1)

## 프로젝트 개요

본 프로젝트는 **PROMPTPEEK** 사이드 채널 공격을 재현하는 보안 연구 프로젝트입니다. 이 공격은 SGLang과 같은 다중 사용자(multi-tenant) LLM 서빙 프레임워크의 **KV 캐시 공유(KV cache sharing)** 메커니즘과 **LPM(Longest Prefix Match) 스케줄링** 정책을 악용하여 다른 사용자의 프롬프트를 토큰 단위로 추론합니다.

**⚠️ 중요 면책조항**: 이 코드는 교육 목적의 보안 연구 프로젝트입니다. 실제 운영 환경에 대한 무단 공격은 불법이며 윤리적으로 문제가 있습니다.

---

## 1. 실험 환경 사양

### 1.1 하드웨어 요구사항
- **GPU**: NVIDIA GPU (A100 80G 권장, RTX 3080 이상 실험 가능)
- **VRAM**: 최소 10GB 이상 (모델 크기에 따라 다름)
- **OS**: Windows 또는 Linux (Mac 제외)

### 1.2 소프트웨어 스택
- **Python**: 3.11.x (엄격히 지정됨)
- **CUDA**: 12.9 이상
- **Docker**: SGLang 서버 컨테이너 실행용
- **LLM 서빙 프레임워크**: SGLang (Docker 이미지: `lmsysorg/sglang:latest`)

### 1.3 대상 및 공격자 모델
- **대상 서버 모델**: Llama-3.1-8B-Instruct (또는 Llama-3.2-3B-Instruct-AWQ)
- **공격자 로컬 모델**: 대상 모델과 동일한 토크나이저를 사용하는 모델 필요
  - 공격자는 후보 토큰 생성을 위해 로컬 LLM이 필요
  - 대상 서버와 동일하거나 호환되는 토크나이저 필수

### 1.4 서버 설정
- **서버 주소**: `http://localhost:30000`
- **최대 배치 크기**: 16 (SGLang 기본값, 논문 설정)
- **KV 캐시 공유**: 활성화 (SGLang 기본 설정)
- **스케줄링 정책**: LPM (Longest Prefix Match) - SGLang 기본값

---

## 2. 핵심 공격 원리

### 2.1 공격 메커니즘
PROMPTPEEK는 다음 원리를 활용합니다:

1. **KV 캐시 공유**: SGLang은 동일한 프롬프트 접두사(prefix)를 가진 요청 간에 KV 캐시를 공유합니다.

2. **LPM 스케줄링**: 서버는 들어오는 요청을 처리할 때, 캐시에 이미 존재하는 접두사와 가장 긴 매치를 가진 요청을 우선 처리합니다.

3. **사이드 채널**: 응답 반환 순서를 관찰하여 어떤 요청이 캐시에 매치되었는지 추론할 수 있습니다.

### 2.2 공격 시나리오
```
[피해자] → 프롬프트 "The secret code is ABC" 전송 → [서버 캐시에 저장]
                                                           ↓
[공격자] → 후보 요청들 전송:                                  ↓
           - "The secret code is A" ← 캐시 매치! 빠르게 반환
           - "The secret code is X" ← 매치 없음, 느리게 반환
           - "The secret code is Y" ← 매치 없음, 느리게 반환
```

---

## 3. 코드 구조 및 설계

### 3.1 디렉토리 구조
```
252_syssec/team_project/
├── src/
│   ├── promptpeek/
│   │   ├── attacker/
│   │   │   └── attacker.py         # 핵심 공격 로직 구현
│   │   └── victim/
│   │       └── victim.py            # 피해자 시뮬레이션
│   ├── faas_server/                 # (필요시) 서버 래퍼
│   └── faas_clients/                # (필요시) 클라이언트 유틸리티
├── docs/                            # 문서 및 실험 결과
├── pyproject.toml                   # 의존성 관리
└── IMPLEMENTATION_GUIDELINES.md     # 본 문서
```

### 3.2 핵심 모듈 설계

#### `src/promptpeek/attacker/attacker.py`
주요 클래스 및 함수:

```python
class PromptPeekAttacker:
    """PROMPTPEEK 공격 실행 클래스"""
    
    def __init__(self, server_url, local_model, config):
        """
        Args:
            server_url: SGLang 서버 주소 (http://localhost:30000)
            local_model: 후보 생성용 로컬 LLM 인스턴스
            config: 공격 설정 (배치 크기, 후보 수 등)
        """
        self.server_url = server_url
        self.local_model = local_model
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def reconstruct_prompt(self) -> str:
        """
        전체 프롬프트 재구성 루프
        1. GPU 메모리 정리
        2. 피해자 프롬프트 캐싱 대기
        3. 다음 토큰 추출 반복
        4. 프롬프트 제거 감지 시 루프 종료
        Returns:
            재구성된 프롬프트 문자열
        """
        pass
    
    async def extract_next_token(self, current_prefix: str) -> Optional[str]:
        """
        다음 토큰 추출 (핵심 기본 연산)
        1. 후보 토큰 생성
        2. [Pre-dummy, Candidates, Post-dummy] 요청 묶음 전송
        3. 응답 순서 관찰하여 매치 토큰 식별
        Args:
            current_prefix: 현재까지 재구성된 프롬프트
        Returns:
            추출된 다음 토큰 또는 None
        """
        pass
    
    async def generate_candidates(self, prefix: str, k: int) -> List[str]:
        """
        로컬 LLM을 사용해 top-k 후보 토큰 생성.
        (고도화): 논문의 시나리오 2/3를 위해 프롬프트 엔지니어링 사용 가능.
        """
        pass
    
    async def send_requests_and_observe_order(self, candidates: List[str], 
                                              prefix: str) -> Optional[str]:
        """
        요청을 특정 순서로 전송하고 응답 순서 관찰
        Returns:
            매치된 토큰 또는 None
        """
        pass
    
    async def clear_gpu_memory(self):
        """
        (논문 Sec. IV-B Optimization)
        '서로 다른(non-identical)' 더미 요청을 전송하여 
        GPU 메모리를 포화시키고 LRU 캐시 제거를 유발.
        """
        pass
    
    async def check_prompt_evicted(self, unmatched_proxy_prompt: str) -> bool:
        """
        (논문 Sec. IV-B Prompt Switching)
        이전에 매치되지 않았던 후보 프롬프트를 '프록시'로 사용하여, 
        해당 프록시가 캐시에서 제거되었는지 확인함으로써 
        피해자의 원본 프롬프트 제거 여부를 간접적으로 추론.
        """
        pass
```

---

## 4. 구현 단계별 가이드

### Phase 1: 기본 인프라 구축
1. **서버 연결 확인**
   - `httpx.AsyncClient`를 사용한 비동기 HTTP 클라이언트 구현
   - localhost:30000 연결 테스트
   - OpenAI 호환 API 엔드포인트 사용: `/v1/chat/completions`

2. **피해자 시뮬레이션**
   - `victim.py`를 확장하여 주기적으로 프롬프트 전송
   - 다양한 프롬프트 템플릿 준비

### Phase 2: 후보 생성 메커니즘
1. **로컬 LLM 통합**
   - Hugging Face Transformers 또는 SGLang 클라이언트 사용
   - 토크나이저 호환성 확인
   - Top-k 토큰 샘플링 구현
   - **(고도화)**: 논문의 시나리오 2(입력 재구성) 또는 3(템플릿 재구성)을 구현할 경우, 단순 top-k 대신 로컬 LLM에 특정 프롬프트(예: '다음 템플릿의 빈칸을 채워: ...')를 주어 후보군의 정확도를 높이는 '프롬프트 엔지니어링' 기법을 적용합니다.

2. **더미 토큰 생성**
   - 낮은 확률의 토큰 선택 (예: " %")
   - 더미 요청 프롬프트 생성

### Phase 3: 토큰 추출 로직 (핵심)
1. **요청 순서 생성**
   ```
   [Pre-candidate 더미 요청 × DUMMY_BATCH_SIZE] (동일한 프롬프트)
   [후보 요청 × CANDIDATE_BATCH_SIZE] (서로 다른 프롬프트)
   [Post-candidate 더미 요청 × DUMMY_BATCH_SIZE] (동일한 프롬프트)
   ```

2. **비동기 요청 전송**
   - `asyncio.gather()` 또는 `asyncio.as_completed()` 사용
   - 모든 요청을 거의 동시에 전송

3. **응답 순서 관찰**
   - 각 응답의 완료 시간 기록
   - **(핵심)** 후보 요청이 Post-dummy보다 먼저 완료되는지 확인
   - 매치 판별 로직 구현 (Figure 6 참조)

### Phase 4: 프롬프트 재구성 루프
1. **초기화 (GPU 메모리 정리)**
   - `clear_gpu_memory`를 호출. 이 함수는 **'서로 다른(non-identical)'** 더미 요청 (예: `prefix + uuid` 또는 `prefix + random_string`)을 서버 최대 용량까지 전송하여 의도적으로 KV 캐시 제거(eviction)를 유발합니다. (논문 Sec. IV-B Optimization)

2. **반복 토큰 추출**
   - 피해자 프롬프트 캐싱 대기 (잠시 `sleep`)
   - 각 반복마다 `extract_next_token()` 호출
   - 재구성된 프롬프트 업데이트
   - 공격 중 사용했던 '매치되지 않은 후보' 중 하나를 프록시로 저장.
   - 종료 조건 확인 (EOS 토큰, 제거 감지)

3. **프롬프트 전환 감지 (`check_prompt_evicted`)**
   - 공격자는 피해자의 프롬프트 상태를 직접 알 수 없습니다.
   - 대신, 이전에 전송했던 **'매치되지 않은 후보 프롬프트'** (예: "The secret code is X")를 `check_prompt_evicted`의 인자로 전달하여 다시 테스트합니다.
   - 만약 이 '프록시' 요청이 캐시에서 제거되었다고 판별되면 (즉, 응답 순서가 다른 더미 요청들보다 늦어지면), 이보다 먼저 캐시된 피해자의 원본 프롬프트도 LRU 정책에 따라 제거되었을 것이라 강하게 **추론**할 수 있습니다. (논문 Sec. IV-B Prompt Switching)

### Phase 5: 최적화 및 평가
1. **성능 튜닝**
   - `DUMMY_BATCH_SIZE` 조정 (권장: `SERVER_MAX_BATCH_SIZE + 4`)
   - `CANDIDATE_BATCH_SIZE` 조정 (권장: 50)
   - 타임아웃 및 재시도 로직

2. **정확도 측정**
   - 재구성된 프롬프트와 실제 프롬프트 비교
   - 토큰 정확도, 문자 정확도 계산

---

## 5. 주요 설정 매개변수

### 5.1 공격 설정
```python
ATTACK_CONFIG = {
    # 서버 설정
    "SERVER_URL": "http://localhost:30000",
    "SERVER_MAX_BATCH_SIZE": 16,  # SGLang 기본값
    
    # 요청 설정
    "DUMMY_BATCH_SIZE": 20,        # SERVER_MAX_BATCH_SIZE + 4
    "CANDIDATE_BATCH_SIZE": 50,    # Top-k 후보 수
    "MAX_TOKENS": 1,               # 각 요청의 최대 출력 길이
    
    # 더미 토큰
    "DUMMY_TOKEN": " %",           # 낮은 확률의 토큰 (토큰 추출 단계용)
    
    # 타이밍
    "CACHE_WAIT_TIME": 5,          # 피해자 프롬프트 캐싱 대기 시간 (초)
    "REQUEST_TIMEOUT": 30,         # HTTP 요청 타임아웃 (초)
    
    # 재구성 설정
    "MAX_PROMPT_LENGTH": 100,      # 최대 재구성 토큰 수
    "RETRY_ON_FAIL": 3,            # 실패 시 재시도 횟수
}
```

### 5.2 로컬 LLM 설정
```python
LOCAL_MODEL_CONFIG = {
    "model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "device": "cuda",
    "load_in_8bit": True,          # 메모리 절약
    "torch_dtype": "auto",
}
```

---

## 6. 의존성 관리

### 6.1 필수 패키지
```toml
[project.dependencies]
httpx = ">=0.28.1"           # 비동기 HTTP 클라이언트
asyncio = "*"                # 비동기 프로그래밍
openai = ">=2.7.1"           # OpenAI 호환 API (선택)
transformers = ">=4.40.0"    # 로컬 LLM 로딩
torch = ">=2.2.0"            # PyTorch
accelerate = ">=0.27.0"      # 모델 로딩 최적화
```

### 6.2 설치 명령
```bash
# PDM 사용 시
pdm add httpx transformers torch accelerate

# 또는 pip
pip install httpx transformers torch accelerate
```

---

## 7. 구현 시 주의사항

### 7.1 비동기 프로그래밍
- **모든 I/O 작업은 비동기로**: `async/await` 사용
- **동시성 제어**: 너무 많은 동시 요청은 서버 과부하 유발
- **타임아웃 처리**: 모든 HTTP 요청에 타임아웃 설정

### 7.2 응답 순서 관찰
- **스트리밍 vs 일괄**: SGLang은 스트리밍 응답 지원
- **정확한 타이밍**: `asyncio.as_completed()` 사용하여 완료 순서 추적
- **태스크 식별**: 각 태스크를 프롬프트와 매핑하는 딕셔너리 유지

### 7.3 로컬 LLM 최적화
- **메모리 관리**: 8bit 양자화 사용 (`load_in_8bit=True`)
- **토크나이저 일치**: 대상 서버와 동일한 토크나이저 필수
- **캐싱**: 로컬 모델의 KV 캐시 활용하여 반복 생성 속도 향상

### 7.4 디버깅 및 로깅
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('promptpeek_attack.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
```

### 7.5 에러 처리
- **네트워크 에러**: 재시도 로직 구현
- **서버 과부하**: 지수 백오프 (exponential backoff)
- **메모리 부족**: 배치 크기 자동 조정

### 7.6 공격 단계별 핵심 로직 (논문 상세)

#### 7.6.1 GPU 메모리 정리 (`clear_gpu_memory`)
- **목적**: 새 프롬프트를 공격하기 전, 기존 캐시를 모두 비워 깨끗한 상태(clean slate)를 만듭니다.
- **로직**: 토큰 추출 단계와는 다르게, **'서로 다른(non-identical)'** 더미 요청을 전송해야 합니다. (예: `f"clear_cache_token_{i}"` 또는 `str(uuid.uuid4())`를 접두사로 사용).
- **이유**: 각 요청이 고유한 캐시를 생성하게 하여 GPU 메모리를 빠르게 점유하고, LRU(Least Recently Used) 정책에 따라 기존에 저장된 모든 캐시(피해자의 프롬프트 포함)를 밀어내도록 강제합니다.

#### 7.6.2 토큰 추출 (`extract_next_token`)
- **목적**: 현재 접두사(`current_prefix`)에 이어지는 다음 토큰 한 개를 식별합니다.
- **로직**: **'동일한(identical)'** 더미 토큰(`DUMMY_TOKEN`)을 사용합니다.
- **이유**: 이 단계의 목적은 메모리를 점유하는 것이 아니라, `DUMMY_BATCH_SIZE` 만큼의 요청으로 서버의 실행 큐를 채우는 것입니다. 동일한 더미 요청을 보내면 캐시가 재사용되므로 *추가적인 GPU 메모리 부담 없이* 대기 큐를 채울 수 있습니다. 이로 인해 뒤따르는 후보(candidate) 요청들이 LPM 스케줄러에 의해 정렬될 수 있습니다.

#### 7.6.3 프롬프트 제거 감지 (`check_prompt_evicted`)
- **목적**: 공격 대상 프롬프트가 캐시에서 제거되었는지 확인하여 공격을 중단하고 새 대상을 찾습니다.
- **로직**: 이전에 전송했던 **'매치되지 않은 후보 프롬프트'** (Unmatched Candidate)를 프록시로 사용합니다. 이 프록시 요청을 다시 '샌드위치' 방식으로 전송하여 응답 순서를 확인합니다.
- **이유**: 프록시 요청은 피해자의 원본 프롬프트보다 *나중에* 캐시되었습니다. 만약 이 프록시마저 LRU 정책에 의해 제거되었다면(응답 순서가 느려짐), 그보다 *먼저* 캐시된 원본 프롬프트는 확실히 제거되었다고 추론할 수 있습니다.

---

## 8. 실험 프로토콜

### 8.1 실험 전 체크리스트
- [ ] SGLang 서버 실행 확인 (`docker ps`)
- [ ] 서버 연결 테스트 (`curl http://localhost:30000/health`)
- [ ] GPU 사용 가능 확인 (`nvidia-smi`)
- [ ] 로컬 LLM 로딩 확인
- [ ] 피해자 스크립트 준비

### 8.2 실험 단계
1. **베이스라인 측정**: 정상 요청의 응답 시간 분포 확인
2. **단일 토큰 추출 테스트**: 알려진 프롬프트로 한 토큰 추출 성공 확인
3. **전체 프롬프트 재구성**: 짧은 프롬프트(5-10 토큰)부터 시작
4. **스케일 업**: 더 긴 프롬프트와 더 많은 후보로 확장
5. **대응책 평가**: (선택) 방어 메커니즘 테스트

### 8.3 성능 지표
- **토큰 정확도**: 올바르게 추출된 토큰 비율
- **문자 정확도**: 문자 단위 정확도 (edit distance)
- **공격 속도**: 토큰당 평균 시간
- **요청 수**: 토큰당 필요한 평균 요청 수

---

## 9. 보안 및 윤리적 고려사항

### 9.1 범위 제한
- **로컬 환경만**: localhost 서버에 대해서만 실험
- **통제된 데이터**: 민감한 실제 데이터 사용 금지
- **문서화**: 모든 실험을 상세히 기록

### 9.2 책임감 있는 공개
- 실험 결과를 공개할 때 악용 가능한 세부 사항 최소화
- SGLang 개발팀에 발견 사항 보고 (책임감 있는 공개)
- 학술적 목적으로만 사용

### 9.3 교육 목적 명시
코드 모든 파일에 다음 주석 포함:
```python
"""
PROMPTPEEK Attack Implementation for Educational Purposes
WARNING: This code is for security research and education only.
Unauthorized use against production systems is illegal and unethical.
"""
```

---

## 10. 참고 자료

### 10.1 논문 핵심 섹션
- **Section 4-A (Figure 5, 6)**: Token Extraction (토큰 추출 기본 연산 및 순서)
- **Section 4-B**: Prompt Reconstruction (프롬프트 재구성 루프)
- **Section 4-B-1 (Prompt Switching)**: `check_prompt_evicted` 로직
- **Section 4-B-2 (Optimization)**: `clear_gpu_memory` 로직
- **Section 6 (Evaluation)**: 실험 설정 및 매개변수 (Figure 10 등)

### 10.2 SGLang 문서
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [API 문서](https://sgl-project.github.io/)
- KV 캐시 공유 메커니즘 설명
- LPM 스케줄러 구현 세부사항

### 10.3 관련 기술
- Prefix caching in LLM serving
- Batch scheduling in GPU inference
- Side-channel attacks in ML systems

---

## 11. 다음 단계

### 즉시 시작
1. `attacker.py`에 `PromptPeekAttacker` 클래스 뼈대 작성
2. 기본 HTTP 클라이언트 및 서버 연결 테스트
3. 로컬 LLM 로딩 및 후보 생성 테스트

### 단기 목표 (1-2주)
- 단일 토큰 추출 성공 (`extract_next_token`)
- 간단한 프롬프트 재구성 (5-10 토큰)
- 기본 로깅 및 디버깅 도구 구축

### 중기 목표 (3-4주)
- 전체 프롬프트 재구성 파이프라인 완성 (`reconstruct_prompt`)
- `clear_gpu_memory` 및 `check_prompt_evicted` 로직 구현
- 정량적 평가 및 문서화

### 장기 목표 (프로젝트 완료)
- 다양한 시나리오에서 실험
- 대응책 구현 및 평가
- 최종 보고서 및 프레젠테이션 준비

---

## 12. 문의 및 트러블슈팅

### 일반적인 문제
1. **"서버 연결 실패"**
   - Docker 컨테이너 상태 확인: `docker ps`
   - 포트 포워딩 확인: `-p 30000:30000`
   - 방화벽 설정 확인

2. **"메모리 부족"**
   - 로컬 LLM 양자화 사용
   - 배치 크기 감소
   - Docker `--shm-size` 증가

3. **"토큰 추출 실패"**
   - `DUMMY_BATCH_SIZE`가 `SERVER_MAX_BATCH_SIZE`보다 큰지 확인
   - 후보 수 증가 (k 값 증가)
   - 타이밍 조정 (요청 간격)

4. **"응답 순서가 예상과 다름"**
   - 서버 부하 확인 (다른 프로세스 종료)
   - 스트리밍 모드 확인
   - LPM 스케줄러 활성화 확인

---

**문서 버전**: 1.1 (논문 로직 보강)
**최종 수정**: 2025-11-05
**작성자**: System Security Team Project
**프로젝트**: ESW7004_41 Team Project - PROMPTPEEK Attack Reproduction

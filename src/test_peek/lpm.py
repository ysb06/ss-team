"""
LPM (Longest Prefix Match) Scheduling Simulator
SGLang의 LPM 스케줄링을 모사하는 HTTP 서버

기본적으로 프롬프트를 포함한 HTTP 요청을 수신하면 Ollama에 해당 프롬프트를 전송하고 응답을 반환
단, 요청을 보내기 전 LPM 스케쥴링을 본 코드에서 수행
KV-Cache는 단순히 딕셔너리로 구현
즉, SGLang의 LPM의 핵심인 가장 긴 매치를 가진 요청을 우선 처리 부분만 구현
"""
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Dict, Tuple
from queue import PriorityQueue, Queue
from dataclasses import dataclass, field
import urllib.request
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
LPM_SERVER_PORT = 9000
OLLAMA_TIMEOUT = 600  # Ollama 요청 타임아웃 (초)
CLIENT_RESPONSE_TIMEOUT = 540  # 클라이언트 응답 대기 타임아웃 (초)


@dataclass(order=True)
class Request:
    """요청 정보를 담는 데이터 클래스"""
    priority: int = field(compare=True)  # 음수값 (긴 매치일수록 우선순위 높음)
    timestamp: float = field(compare=True)
    prompt: str = field(compare=False)
    request_id: str = field(compare=False)
    response_queue: Queue = field(compare=False)
    cancelled: threading.Event = field(compare=False, default_factory=threading.Event)


class KVCache:
    """KV 캐시를 시뮬레이션하는 클래스"""
    
    def __init__(self):
        self.cache: Dict[str, float] = {}  # {prompt: cached_timestamp}
        self.lock = threading.Lock()
    
    def add(self, prompt: str):
        """캐시에 프롬프트 추가"""
        with self.lock:
            self.cache[prompt] = time.time()
            logger.info(f"[CACHE] Added: '{prompt[:50]}...'")
    
    def find_longest_prefix_match(self, prompt: str) -> Tuple[str, int]:
        """
        주어진 프롬프트와 가장 긴 공통 접두사를 가진 캐시 엔트리 찾기
        Returns: (matched_prefix, match_length)
        """
        with self.lock:
            longest_match = ""
            max_length = 0
            
            for cached_prompt in self.cache.keys():
                # 두 문자열의 공통 접두사 길이 계산
                common_prefix_len = 0
                for i in range(min(len(cached_prompt), len(prompt))):
                    if cached_prompt[i] == prompt[i]:
                        common_prefix_len += 1
                    else:
                        break
                
                if common_prefix_len > max_length:
                    max_length = common_prefix_len
                    longest_match = cached_prompt[:common_prefix_len]
            
            return longest_match, max_length
    
    def clear(self):
        """캐시 초기화"""
        with self.lock:
            self.cache.clear()
            logger.info("[CACHE] Cleared")


class LPMScheduler:
    """LPM 스케줄링을 수행하는 클래스"""
    
    def __init__(self, kv_cache: KVCache, num_workers: int = 4):
        self.kv_cache = kv_cache
        self.request_queue = PriorityQueue()
        self.worker_threads = []
        self.num_workers = num_workers
        self.running = True
    
    def submit_request(self, prompt: str, request_id: str) -> tuple[Queue, Request]:
        """요청을 스케줄러에 제출"""
        # LPM: 가장 긴 prefix 매치를 찾아서 우선순위 계산
        matched_prefix, match_length = self.kv_cache.find_longest_prefix_match(prompt)
        
        # 우선순위: 매치 길이가 길수록 높은 우선순위 (음수 사용)
        priority = -match_length
        
        response_queue = Queue()
        request = Request(
            priority=priority,
            timestamp=time.time(),
            prompt=prompt,
            request_id=request_id,
            response_queue=response_queue
        )
        
        if match_length > 0:
            logger.debug(f"[SCHEDULE] Request '{request_id}' - Cache HIT (match_length={match_length})")
        else:
            logger.debug(f"[SCHEDULE] Request '{request_id}' - Cache MISS")
        
        self.request_queue.put(request)
        
        return response_queue, request
    
    def start_worker(self):
        """워커 스레드 시작 (멀티 워커)"""
        for i in range(self.num_workers):
            worker_thread = threading.Thread(target=self._process_requests, args=(i,), daemon=True)
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        logger.info(f"[SCHEDULER] Started {self.num_workers} worker threads")
    
    def _process_requests(self, worker_id: int):
        """요청을 우선순위 순서대로 처리 (멀티 워커)"""
        logger.info(f"[WORKER-{worker_id}] Started")
        while self.running:
            try:
                # 우선순위 큐에서 요청 가져오기 (우선순위 높은 것부터)
                request = self.request_queue.get(timeout=1.0)
                
                # 취소된 요청은 건너뜀
                if request.cancelled.is_set():
                    logger.info(f"[WORKER-{worker_id}] Request '{request.request_id}' cancelled, skipping")
                    self.request_queue.task_done()
                    continue
                
                logger.debug(f"[WORKER-{worker_id}] Processing '{request.request_id}' - Priority={request.priority}")
                
                # Ollama에 요청 전송
                response_text = self._send_to_ollama(request.prompt)
                
                # 응답을 요청자에게 전달
                request.response_queue.put(response_text)
                
                # 캐시에 추가
                self.kv_cache.add(request.prompt)
                
                self.request_queue.task_done()
                
            except:
                # 타임아웃이나 다른 예외는 무시하고 계속 대기
                continue
    
    def _send_to_ollama(self, prompt: str) -> str:
        """Ollama에 실제 요청 전송"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            req = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
                resp_body = response.read().decode('utf-8')
                resp_json = json.loads(resp_body)
                answer = resp_json.get("response", "")
                return answer
                
        except Exception as e:
            logger.warning(f"[OLLAMA] Error: {e}")
            return f"Error: {str(e)}"


# Global instances
kv_cache = KVCache()
scheduler = LPMScheduler(kv_cache)


class LPMRequestHandler(BaseHTTPRequestHandler):
    """LPM 서버의 HTTP 요청 핸들러"""
    
    def log_message(self, format, *args):
        """로그 메시지를 logger로 리다이렉트"""
        logger.debug(f"HTTP: {format % args}")
    
    def do_POST(self):
        """POST 요청 처리"""
        if self.path == "/generate":
            request_obj = None
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                post_data = self.rfile.read(content_length).decode('utf-8')
                
                if not post_data:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"No input provided")
                    return
                
                request_id = f"req_{int(time.time() * 1000000)}"
                
                # 스케줄러에 요청 제출
                response_queue, request_obj = scheduler.submit_request(post_data, request_id)
                
                # 응답 대기 (클라이언트 타임아웃보다 약간 짧게)
                response_text = response_queue.get(timeout=CLIENT_RESPONSE_TIMEOUT)
                
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(response_text.encode('utf-8'))
                
            except BrokenPipeError:
                logger.debug(f"[HTTP] Client disconnected (BrokenPipe)")
                if request_obj:
                    request_obj.cancelled.set()
                return
            except ConnectionResetError:
                logger.debug(f"[HTTP] Client disconnected (ConnectionReset)")
                if request_obj:
                    request_obj.cancelled.set()
                return
            except Exception as e:
                logger.error(f"[HTTP] Error processing request: {type(e).__name__}: {e}")
                if request_obj:
                    request_obj.cancelled.set()
                try:
                    self.send_response(500)
                    self.end_headers()
                except:
                    pass  # 클라이언트가 이미 연결을 끊은 경우
                return
        
        elif self.path == "/flush_cache":
            kv_cache.clear()
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Cache flushed")
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        """GET 요청 처리"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"LPM Server is running")
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = LPM_SERVER_PORT):
    """LPM 서버 실행"""
    # 스케줄러 워커 시작
    scheduler.start_worker()
    
    server_address = ('', port)
    # ThreadingHTTPServer로 변경하여 멀티스레드 요청 처리
    httpd = ThreadingHTTPServer(server_address, LPMRequestHandler)
    logger.info(f"LPM Server starting on port {port}...")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"Ollama Model: {OLLAMA_MODEL}")
    logger.info(f"Number of worker threads: {scheduler.num_workers}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        scheduler.running = False
        httpd.shutdown()


if __name__ == "__main__":
    run_server()

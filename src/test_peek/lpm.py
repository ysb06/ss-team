"""
LPM (Longest Prefix Match) Scheduling Simulator
HTTP server simulating SGLang's LPM scheduling

Basically, when receiving HTTP requests containing prompts, sends the prompts to Ollama and returns responses
However, LPM scheduling is performed in this code before sending requests
KV-Cache is simply implemented as a dictionary
In other words, only the core part of SGLang's LPM - prioritizing requests with longest match - is implemented
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
OLLAMA_TIMEOUT = 720  # Ollama request timeout (seconds)
CLIENT_RESPONSE_TIMEOUT = 725  # Client response wait timeout (seconds)


@dataclass(order=True)
class Request:
    """Data class containing request information"""
    priority: int = field(compare=True)  # Negative value (longer match = higher priority)
    timestamp: float = field(compare=True)
    prompt: str = field(compare=False)
    request_id: str = field(compare=False)
    response_queue: Queue = field(compare=False)
    cancelled: threading.Event = field(compare=False, default_factory=threading.Event)


class KVCache:
    """Class simulating KV cache"""
    
    def __init__(self):
        self.cache: Dict[str, float] = {}  # {prompt: cached_timestamp}
        self.lock = threading.Lock()
    
    def add(self, prompt: str):
        """Add prompt to cache"""
        with self.lock:
            self.cache[prompt] = time.time()
            logger.info(f"[CACHE] Added: '{prompt[:50]}...' (Total cached: {len(self.cache)})")
    
    def find_longest_prefix_match(self, prompt: str) -> Tuple[str, int]:
        """
        Find cache entry with longest prefix match for given prompt
        Unidirectional matching: Check if cached prompt is a prefix of new request
        Returns: (matched_prefix, match_length)
        """
        with self.lock:
            longest_match = ""
            max_length = 0
            
            logger.debug(f"[CACHE] Finding LPM for prompt: '{prompt[:50]}...' (Cache size: {len(self.cache)})")
            
            for cached_prompt in self.cache.keys():
                # Unidirectional matching: Check if cached_prompt starts with prompt
                # (Check if long cached prompt has new request as prefix)
                if cached_prompt.startswith(prompt):
                    if len(cached_prompt) > max_length:
                        max_length = len(cached_prompt)
                        longest_match = cached_prompt
            
            if max_length > 0:
                logger.info(f"[CACHE] LPM HIT: match_length={max_length}, matched='{longest_match[:50]}...'")
            else:
                logger.info(f"[CACHE] LPM MISS: No prefix match found")
            
            return longest_match, max_length
    
    def clear(self):
        """Clear cache"""
        with self.lock:
            cache_size = len(self.cache)
            self.cache.clear()
            logger.info(f"[CACHE] Cleared {cache_size} entries")
    
    def dump_state(self):
        """Dump current cache state (for debugging)"""
        with self.lock:
            logger.info(f"[CACHE] === Cache State Dump (Total: {len(self.cache)}) ===")
            for i, (prompt, timestamp) in enumerate(self.cache.items(), 1):
                logger.info(f"[CACHE]   {i}. '{prompt[:60]}...' (cached at {timestamp:.3f})")
            logger.info(f"[CACHE] === End of Cache Dump ===")


class LPMScheduler:
    """Class performing LPM scheduling"""
    
    def __init__(self, kv_cache: KVCache, num_workers: int = 4):
        self.kv_cache = kv_cache
        self.request_queue = PriorityQueue()
        self.worker_threads = []
        self.num_workers = num_workers
        self.running = True
        self.active_requests = []  # Track currently processing requests
        self.active_requests_lock = threading.Lock()
    
    def submit_request(self, prompt: str, request_id: str) -> tuple[Queue, Request]:
        """Submit request to scheduler"""
        # LPM: Find longest prefix match and calculate priority
        matched_prefix, match_length = self.kv_cache.find_longest_prefix_match(prompt)
        
        # Priority: longer match length = higher priority (use negative value)
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
            logger.info(f"[SCHEDULE] Request '{request_id}' - Cache HIT (match_length={match_length}, priority={priority})")
        else:
            logger.info(f"[SCHEDULE] Request '{request_id}' - Cache MISS (priority={priority})")
        
        logger.debug(f"[SCHEDULE] Queuing request '{request_id}' with prompt: '{prompt[:50]}...'")
        self.request_queue.put(request)
        
        return response_queue, request
    
    def start_worker(self):
        """Start worker threads (multi-worker)"""
        for i in range(self.num_workers):
            worker_thread = threading.Thread(target=self._process_requests, args=(i,), daemon=True)
            worker_thread.start()
            self.worker_threads.append(worker_thread)
        logger.info(f"[SCHEDULER] Started {self.num_workers} worker threads")
    
    def _process_requests(self, worker_id: int):
        """Process requests in priority order (multi-worker)"""
        logger.info(f"[WORKER-{worker_id}] Started")
        while self.running:
            try:
                # 우선순위 큐에서 요청 가져오기 (우선순위 높은 것부터)
                request = self.request_queue.get(timeout=1.0)
                
                # 취소된 요청은 건너뜀
                if request.cancelled.is_set():
                    logger.info(f"[WORKER-{worker_id}] Request '{request.request_id}' cancelled, skipping")
                    request.response_queue.put("Error: Request cancelled")
                    self.request_queue.task_done()
                    continue
                
                # 활성 요청 목록에 추가
                with self.active_requests_lock:
                    self.active_requests.append(request)
                
                start_time = time.time()
                logger.info(f"[WORKER-{worker_id}] ⚡ START Processing '{request.request_id}' - Priority={request.priority}, Prompt: '{request.prompt[:50]}...'")
                
                # LPM 체크: 캐시에 존재하는지 확인
                matched_prefix, match_length = self.kv_cache.find_longest_prefix_match(request.prompt)
                is_cache_hit = match_length > 0
                
                if is_cache_hit:
                    # 캐시 히트: 즉시 응답 (Ollama 호출 없음)
                    logger.info(f"[WORKER-{worker_id}] 🚀 CACHE HIT! Responding immediately (match_length={match_length})")
                    if not request.cancelled.is_set():
                        # 더미 응답 즉시 반환 (실제 SGLang은 캐시된 토큰 재사용)
                        response_text = "[CACHED] Response based on cached KV"
                        request.response_queue.put(response_text)
                    else:
                        request.response_queue.put("Error: Request cancelled")
                else:
                    # 캐시 미스: Ollama 처리 필요
                    logger.info(f"[WORKER-{worker_id}] ❄️  CACHE MISS - Processing with Ollama")
                    
                    # 캐시에 먼저 추가 (다음 요청을 위해)
                    if not request.cancelled.is_set():
                        cache_add_time = time.time()
                        self.kv_cache.add(request.prompt)
                        logger.info(f"[WORKER-{worker_id}] ✓ Cached '{request.request_id}' at {cache_add_time:.3f}")
                    
                    # Ollama에 요청 전송 (취소 체크 포함)
                    if not request.cancelled.is_set():
                        ollama_start = time.time()
                        logger.debug(f"[WORKER-{worker_id}] Sending to Ollama: '{request.request_id}'")
                        response_text = self._send_to_ollama(request.prompt)
                        ollama_duration = time.time() - ollama_start
                        logger.debug(f"[WORKER-{worker_id}] Ollama responded in {ollama_duration:.3f}s")
                        
                        # Check cancellation again before sending response
                        if not request.cancelled.is_set():
                            # Send response to requester
                            request.response_queue.put(response_text)
                        else:
                            request.response_queue.put("Error: Request cancelled")
                    else:
                        request.response_queue.put("Error: Request cancelled")
                
                # Remove from active request list
                with self.active_requests_lock:
                    if request in self.active_requests:
                        self.active_requests.remove(request)
                
                end_time = time.time()
                total_duration = end_time - start_time
                remaining_requests = self.request_queue.qsize()
                logger.info(f"[WORKER-{worker_id}] ✓ DONE '{request.request_id}' in {total_duration:.3f}s (Queue: {remaining_requests} remaining)")
                
                self.request_queue.task_done()
                
            except:
                # Ignore timeout and other exceptions, continue waiting
                continue
    
    def flush_all(self):
        """Abort all requests and clear cache"""
        logger.info("[SCHEDULER] Flushing all requests and cache...")
        
        # 1. Set cancel flag for all currently processing requests
        with self.active_requests_lock:
            for request in self.active_requests:
                request.cancelled.set()
            active_count = len(self.active_requests)
        
        # 2. Remove all requests from queue and cancel them
        cancelled_queue_count = 0
        while not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                request.cancelled.set()
                request.response_queue.put("Error: Request cancelled by flush")
                self.request_queue.task_done()
                cancelled_queue_count += 1
            except:
                break
        
        # 3. Clear cache
        self.kv_cache.clear()
        
        logger.info(f"[SCHEDULER] Flushed: {active_count} active requests, {cancelled_queue_count} queued requests")
    
    def _send_to_ollama(self, prompt: str) -> str:
        """Send actual request to Ollama"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 16
            }
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
    """HTTP request handler for LPM server"""
    
    def log_message(self, format, *args):
        """Redirect log messages to logger"""
        logger.debug(f"HTTP: {format % args}")
    
    def do_POST(self):
        """Handle POST request"""
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
                
                # Submit request to scheduler
                response_queue, request_obj = scheduler.submit_request(post_data, request_id)
                
                # Wait for response (slightly shorter than client timeout)
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
                    pass  # Client already disconnected
                return
        
        elif self.path == "/flush_cache":
            scheduler.flush_all()
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"All requests cancelled and cache flushed")
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_GET(self):
        """Handle GET request"""
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"LPM Server is running")
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port: int = LPM_SERVER_PORT):
    """Run LPM server"""
    # Start scheduler worker
    scheduler.start_worker()
    
    server_address = ('', port)
    # Change to ThreadingHTTPServer for multi-threaded request handling
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

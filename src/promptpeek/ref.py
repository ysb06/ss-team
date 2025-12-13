import requests
import json


def send_to_sglang(prompt: str, server_url: str = "http://192.168.1.8:30000") -> dict:
    """
    SGLang 서버에 텍스트를 보내고 응답을 받습니다.
    
    Args:
        prompt: 전송할 텍스트 프롬프트
        server_url: SGLang 서버 URL (기본값: http://192.168.1.8:30000)
    
    Returns:
        서버로부터 받은 응답 딕셔너리
    """
    # SGLang API 엔드포인트
    endpoint = f"{server_url}/generate"
    
    # 요청 데이터 구성
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": 512,
            "temperature": 0.7,
        }
    }
    
    try:
        # POST 요청 전송
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # 응답 확인
        response.raise_for_status()
        
        # JSON 응답 파싱
        result = response.json()
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"❌ 서버 연결 실패: {server_url}")
        raise
    except requests.exceptions.Timeout:
        print(f"⏱️ 요청 시간 초과")
        raise
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 중 오류 발생: {e}")
        raise
    except json.JSONDecodeError:
        print(f"❌ 응답 JSON 파싱 실패")
        raise


def main():
    """메인 함수"""
    print("=== SGLang 서버 테스트 ===\n")
    
    # 테스트 프롬프트
    test_prompt = "Hello! Please introduce yourself."
    
    print(f"📤 전송 프롬프트: {test_prompt}")
    print(f"🌐 서버 주소: 192.168.1.8:30000\n")
    
    try:
        # SGLang 서버에 요청 전송
        result = send_to_sglang(test_prompt)
        
        # 결과 출력
        print("✅ 응답 수신 성공!\n")
        print("📥 응답 내용:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return 1
    
    return 0
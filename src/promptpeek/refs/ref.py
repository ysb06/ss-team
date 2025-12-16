import requests
import json


def send_to_sglang(prompt: str, server_url: str = "http://192.168.1.8:30000") -> dict:
    """
    Send text to SGLang server and receive response.
    
    Args:
        prompt: Text prompt to send
        server_url: SGLang server URL (default: http://192.168.1.8:30000)
    
    Returns:
        Response dictionary received from server
    """
    # SGLang API endpoint
    endpoint = f"{server_url}/generate"
    
    # Configure request data
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": 512,
            "temperature": 0.7,
        }
    }
    
    try:
        # Send POST request
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Check response
        response.raise_for_status()
        
        # Parse JSON response
        result = response.json()
        return result
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Server connection failed: {server_url}")
        raise
    except requests.exceptions.Timeout:
        print(f"⏱️ Request timeout")
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
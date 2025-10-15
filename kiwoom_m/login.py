import requests
import json
from config import app_key,app_secret,host_url
# 접근토큰 발급
def fn_au10001():
	# 1. 요청할 API URL
	#host = 'https://mockapi.kiwoom.com' # 모의투자
	#host = 'https://api.kiwoom.com' # 실전투자
	endpoint = '/oauth2/token'
	url =  host_url + endpoint

	# 2. header 데이터
	headers = {
		'Content-Type': 'application/json;charset=UTF-8', # 컨텐츠타입
	}
	# 1. 요청 데이터
	data = {
		'grant_type': 'client_credentials',  # grant_type
		'appkey': app_key,  # 앱키
		'secretkey': app_secret,  # 시크릿키
	}
	# 3. http POST 요청
	response = requests.post(url, headers=headers, json=data)
	token =response.json().get('token')
	return token
	# 4. 응답 상태 코드와 데이터 출력
	# print('Code:', response.status_code)
	# print('Header:', json.dumps({key: response.headers.get(key) for key in ['next-key', 'cont-yn', 'api-id']}, indent=4, ensure_ascii=False))
	# print('Body:', json.dumps(response.json(), indent=4, ensure_ascii=False))  # JSON 응답을 파싱하여 출력

# 실행 구간
if __name__ == '__main__':
	# 2. API 실행
	token =fn_au10001()
	print("토큰: ",token)
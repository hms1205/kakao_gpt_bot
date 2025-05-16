import pyautogui
import time
from PIL import Image, ImageChops
import numpy as np
from gpt4all import GPT4All
from paddleocr import PaddleOCR
import re
import pyperclip
import cv2

# GPT 모델 로딩
model = GPT4All(
    model_path="./models",
    # model_name="EEVE-Korean-Instruct-7B-v2.0-Preview.Q8_0",
    model_name="llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M",
    device="cuda",
    n_threads=8,
    allow_download=False,
)

# PaddleOCR 초기화 (한번만 로드)
ocr = PaddleOCR(lang="korean", use_angle_cls=False)

# 캡처 영역 설정 (카톡창 메시지 영역)
CAPTURE_REGION = (1600, 780, 300, 170)


def capture_chat():
    """화면의 특정 영역을 캡처합니다"""
    screenshot = pyautogui.screenshot(region=CAPTURE_REGION)
    screenshot.save("last_capture.png")
    return screenshot


def is_different(img1, img2, threshold=1000):
    """두 이미지 간의 차이를 감지합니다 (임계값 이상이면 다른 것으로 판단)"""
    diff = ImageChops.difference(img1, img2)
    diff_array = np.array(diff)
    diff_sum = np.sum(diff_array)
    return diff_sum > threshold


def extract_text_with_paddle(image_path):
    """PaddleOCR을 사용하여 이미지에서 텍스트 추출 (흰색 말풍선만)"""
    # 이미지 읽기
    img = cv2.imread(image_path)

    # BGR → HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 흰색 범위 설정 (채도 낮고 밝기 높음)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    # 흰색 마스크 생성
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 마스크된 흰색 영역만 추출
    white_only = cv2.bitwise_and(img, img, mask=mask)

    # 저장 (디버깅용)
    cv2.imwrite("filtered_white_only.png", white_only)

    # OCR 입력을 위해 그레이스케일 변환
    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)

    # 필요 시 임계처리 (선택사항)
    # _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # OCR 실행
    result = ocr.ocr(gray, cls=False)

    if not result or not result[0]:
        return ""

    # 텍스트 추출
    text_parts = [line[1][0] for line in result[0]]
    full_text = " ".join(text_parts).strip()

    # 불필요한 문자 제거 (숫자 제외)
    clean_text = re.sub(r"[^\sA-Za-z가-힣0-9.?!,]", "", full_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    print(f"원본 OCR 결과: {text_parts}")
    print(f"정리된 텍스트: {clean_text}")

    return clean_text


# def extract_text_with_paddle(image_path):
#     """PaddleOCR을 사용하여 이미지에서 텍스트 추출"""
#     result = ocr.ocr(image_path, cls=False)

#     # 결과가 비어있는지 확인
#     if not result or not result[0]:
#         return ""

#     # 감지된 모든 텍스트 결합
#     text_parts = []
#     for line in result[0]:
#         text_parts.append(line[1][0])  # OCR 결과에서 텍스트 부분만 추출

#     # 텍스트 정리
#     full_text = " ".join(text_parts).strip()
#     # 한글, 영문, 숫자, 공백, 일부 문장부호만 유지
#     # clean_text = re.sub(r"[^\w\s.?!,가-힣]", "", full_text)
#     clean_text = re.sub(r"[^\sA-Za-z가-힣.?!,]", "", full_text)
#     # 여러 공백을 하나로 치환
#     clean_text = re.sub(r"\s+", " ", clean_text).strip()

#     print(f"원본 OCR 결과: {text_parts}")
#     print(f"정리된 텍스트: {clean_text}")

#     return clean_text


def generate_response(prompt):
    """GPT 모델을 사용하여 응답 생성"""
    instruction = (
        "다음 질문에 대해 최대한 100자 이하로 최대한 짧게 말 해줘. 추가질문 금지. "
        f"질문: {prompt}"
    )
    print(f"프롬프트: {instruction}")

    with model.chat_session():
        response = model.generate(
            prompt=instruction,
            max_tokens=128,
        )
        response = response.strip()
        send_to_kakao(response)
    return True


def send_to_kakao(response):
    """카카오톡에 응답 전송 (클립보드 사용)"""
    # 응답을 클립보드에 복사
    pyperclip.copy(response)

    # 메시지 창에 포커스
    # pyautogui.click(CAPTURE_REGION[0] + CAPTURE_REGION[2]//2,
    #                CAPTURE_REGION[1] + CAPTURE_REGION[3] + 50)

    # 붙여넣기 단축키 사용 (Ctrl+V)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)  # 약간의 지연 추가
    pyautogui.press("enter")


def main():
    print("🔄 카카오톡 감지 시작 (5초마다 확인)")

    prev_img = capture_chat()
    prev_text = extract_text_with_paddle("last_capture.png")
    print(f"초기 상태: {prev_text}")

    last_response_time = 0  # 마지막 응답 시간
    min_response_interval = 5  # 최소 응답 간격(초)

    while True:
        try:
            time.sleep(5)

            curr_img = capture_chat()

            if is_different(prev_img, curr_img):
                # 텍스트 추출
                curr_text = extract_text_with_paddle("last_capture.png")

                # 이미지는 달라도 텍스트가 같으면 무시
                if curr_text != prev_text and curr_text:
                    print(f"📩 새로운 메시지: {curr_text}")

                    # 한글 문자가 있는지 확인
                    has_korean = any(
                        ord(c) >= ord("가") and ord(c) <= ord("힣") for c in curr_text
                    )

                    # 응답 간격 확인 (너무 빠른 응답 방지)
                    current_time = time.time()
                    enough_time_passed = (
                        current_time - last_response_time
                    ) > min_response_interval

                    if has_korean and enough_time_passed:
                        # GPT 응답 생성
                        generate_response(curr_text)
                        # print(f"🤖 GPT 응답: {reply}")

                        # 응답 전송
                        # send_to_kakao(reply)

                        # 마지막 응답 시간 업데이트
                        last_response_time = current_time
                    elif not has_korean:
                        print("⚠️ 한글이 없는 메시지입니다")
                    elif not enough_time_passed:
                        print("⏱️ 응답 간격이 너무 짧습니다")

                    prev_text = curr_text

                prev_img = curr_img
            else:
                print("⏳ 변화 없음")

        except Exception as e:
            print(f"오류 발생: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()

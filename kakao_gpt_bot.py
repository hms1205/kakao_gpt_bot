import pyautogui
import time
from PIL import Image, ImageChops
import numpy as np
from gpt4all import GPT4All
from paddleocr import PaddleOCR
import re
import pyperclip
import cv2

model = GPT4All(
    model_path="./models",
    # model_name="EEVE-Korean-Instruct-7B-v2.0-Preview.Q8_0",
    model_name="llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M",
    device="cuda",
    n_threads=8,
    allow_download=False,
)

ocr = PaddleOCR(lang="korean", use_angle_cls=False)

# 캡처 영역 설정
CAPTURE_REGION = (1600, 780, 300, 170)


def capture_chat():
    screenshot = pyautogui.screenshot(region=CAPTURE_REGION)
    screenshot.save("last_capture.png")
    return screenshot


def is_different(img1, img2, threshold=1000):
    diff = ImageChops.difference(img1, img2)
    diff_array = np.array(diff)
    diff_sum = np.sum(diff_array)
    return diff_sum > threshold


def extract_text_with_paddle(image_path):
    img = cv2.imread(image_path)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)

    white_only = cv2.bitwise_and(img, img, mask=mask)

    # 저장 (디버깅용)
    cv2.imwrite("filtered_white_only.png", white_only)

    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)

    # 필요 시 임계처리 (선택사항)
    # _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    result = ocr.ocr(gray, cls=False)

    if not result or not result[0]:
        return ""

    text_parts = [line[1][0] for line in result[0]]
    full_text = " ".join(text_parts).strip()

    clean_text = re.sub(r"[^\sA-Za-z가-힣0-9.?!,]", "", full_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    print(f"원본 OCR 결과: {text_parts}")
    print(f"정리된 텍스트: {clean_text}")

    return clean_text


def generate_response(prompt):
    instruction = (
        "다음 질문에 대해 최대한 100자 이하로 최대한 짧게 말 해줘. 추가질문 금지. "
        f"질문: {prompt}"
    )

    with model.chat_session():
        response = model.generate(
            prompt=instruction,
            max_tokens=128,
        )
        response = response.strip()
        send_to_kakao(response)
    return True


def send_to_kakao(response):
    pyperclip.copy(response)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.press("enter")


def main():
    check_time = 5
    print(f"감지 시작 ({check_time}초마다 확인)")

    prev_img = capture_chat()
    prev_text = extract_text_with_paddle("last_capture.png")
    print(f"초기 상태: {prev_text}")

    last_response_time = 0  # 마지막 응답 시간
    min_response_interval = 5  # 최소 응답 간격(초)

    while True:
        try:
            time.sleep(check_time)

            curr_img = capture_chat()

            if is_different(prev_img, curr_img):
                curr_text = extract_text_with_paddle("last_capture.png")

                if curr_text != prev_text and curr_text:
                    print(f"새로운 메시지: {curr_text}")

                    has_korean = any(
                        ord(c) >= ord("가") and ord(c) <= ord("힣") for c in curr_text
                    )

                    # 응답 간격 확인
                    current_time = time.time()
                    enough_time_passed = (
                        current_time - last_response_time
                    ) > min_response_interval

                    if has_korean and enough_time_passed:
                        generate_response(curr_text)
                        last_response_time = current_time
                    elif not has_korean:
                        print("한글이 없는 메시지입니다")
                    elif not enough_time_passed:
                        print("응답 간격이 너무 짧습니다")

                    prev_text = curr_text

                prev_img = curr_img
            else:
                print("변화 없음")

        except Exception as e:
            print(f"오류 발생: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()

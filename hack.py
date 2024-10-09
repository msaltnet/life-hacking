import subprocess
import keyboard
import easyocr
import cv2
import time
from collections import Counter

def load_model():
    print("OCR 모델 로딩 중...")
    reader = easyocr.Reader(['ko'], gpu=False)  # 한국어 OCR 모델 로딩
    print("OCR 모델 로딩 완료!")
    return reader

def run_adb_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

def get_screen_resolution():
    output = run_adb_command(["adb", "shell", "wm", "size"])
    resolution = output.split()[-1]  # "Physical size: 1080x1920" 형식에서 "1080x1920" 추출
    width, height = map(int, resolution.split('x'))
    print(f"디바이스의 화면 해상도 : {width}x{height}")
    return width, height

def check_device_connected():
    output = run_adb_command(["adb", "devices"])
    devices = output.splitlines()
    if len(devices) < 2 or not devices[1].strip():
        print("연결된 디바이스가 없어서 종료합니다!")
        return False
    print("디바이스 연결을 확인하였습니다.")
    return True

def capture_screenshot():
    run_adb_command(["adb", "shell", "screencap", "/sdcard/screenshot.png"])
    run_adb_command(["adb", "pull", "/sdcard/screenshot.png"])
    return "screenshot.png"

def resize_image_aspect_ratio(image_path, max_length=800):
    image = cv2.imread(image_path)

    if max_length <= 0:
        return image, 1

    height, width = image.shape[:2]
    if width > height:
        scale_ratio = max_length / width
    else:
        scale_ratio = max_length / height

    new_size = (int(width * scale_ratio), int(height * scale_ratio))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image, scale_ratio

# OCR을 사용하여 텍스트 추출
def extract_text_from_image(reader, image):
    result = reader.readtext(image)
    if result:
        print("검출된 텍스트:")
        for (bbox, text, prob) in result:
            print(f"Text: {text}, Position: {bbox}, Confidence: {prob}")
        return result
    else:
        return []

def find_correct_text(result):
    # 3개 이상 존재하는 테스트가 정상이닷!
    text_list = []

    for (bbox, text, prob) in result:
        if prob > 0.5:
            text_list.append(text)

    element_counts = Counter(text_list)
    for element, count in element_counts.items():
        if count >= 3:
            return element

    print("정상 텍스트를 찾을 수 없습니다!")

def touch_different_text_position(result, scale_ratio):
    if result:
        correct_text = find_correct_text(result)
        size = len(result)
        bbox = None
        print(f"정상 텍스트: {correct_text}")
        for i, value in enumerate(result):
            print(f"Text: {value[1]}, in {i}")
            if value[1] != correct_text and (i > 0 and result[i - 1][1] == correct_text) and (i < size - 1 and result[i + 1][1] == correct_text):
                bbox = value[0]
                break

        if bbox is None:
            print("실패!")
            return

        x_center = int((bbox[0][0] + bbox[2][0]) / 2 / scale_ratio)  # 보정된 x 좌표
        y_center = int((bbox[0][1] + bbox[2][1]) / 2 / scale_ratio)  # 보정된 y 좌표
        print(f"터치 좌표: {x_center}, {y_center}")
        run_adb_command(["adb", "shell", "input", "tap", str(x_center), str(y_center)])
    else:
        print("텍스트가 없습니다!")

def main():
    print("프로그램 실행 중...")
    if not check_device_connected():
        return
    get_screen_resolution()

    reader = load_model()
    print("'Space' 또는 'Enter'를 눌러서 시작하세요!")
    print("그 외 키를 누르면 종료됩니다.")

    while True:
        if keyboard.is_pressed("space") or keyboard.is_pressed("enter"):
            start_time = time.time()

            print("스크린샷 촬영 후 OCR 처리 중...")
            image_path = capture_screenshot()
            resized_image, scale_ratio = resize_image_aspect_ratio(image_path)
            result = extract_text_from_image(reader, resized_image)
            touch_different_text_position(result, scale_ratio)

            print(f"완료! 소요시간: {time.time() - start_time:.2f} 초")

        elif keyboard.read_event().name not in ["space", "enter"]:
            print("프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    main()

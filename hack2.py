import subprocess
import keyboard
import cv2
import time
import numpy as np

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

def find_most_different_image_using_matchTemplate(cropped_images, rect_coords):
    # 모든 이미지를 동일한 크기로 리사이징 (첫 번째 이미지의 크기로)
    target_size = cropped_images[0].shape[:2]  # 첫 번째 이미지의 크기
    resized_images = [cv2.resize(img, (target_size[1], target_size[0])) for img in cropped_images]

    # 이미지 간 유사도를 저장할 배열
    similarity_matrix = np.zeros((len(resized_images), len(resized_images)))

    # 각 이미지에 대해 다른 이미지들과의 유사도 비교
    for i, img1 in enumerate(resized_images):
        for j, img2 in enumerate(resized_images):
            if i == j:
                similarity_matrix[i][j] = 1  # 자기 자신과의 비교는 유사도가 1
            else:
                # 템플릿 매칭을 이용한 유사도 계산
                result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
                similarity_matrix[i][j] = np.max(result)  # 최대 유사도 값

    # 각 이미지의 평균 유사도 계산
    average_similarities = np.mean(similarity_matrix, axis=1)

    # 가장 유사하지 않은 이미지 찾기 (유사도 평균이 가장 낮은 이미지)
    most_different_index = np.argmin(average_similarities)
    cv2.imwrite(f"{str(time.time())[:10]}-most_different.jpg", cropped_images[most_different_index])

    # 가장 다른 이미지의 좌표 반환
    most_different_coords = rect_coords[most_different_index]

    print(f"가장 다른 이미지의 좌표: {most_different_coords}")
    return most_different_coords

def get_answer_position(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 히스토그램 균일화 적용 (대비 향상)
    equalized = cv2.equalizeHist(gray)
    cv2.imwrite(f"{str(time.time())[:10]}-equalized.jpg", equalized)

    # Canny 엣지 검출
    edges = cv2.Canny(equalized, 50, 150)

    # Contours 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선 면적 필터링
    min_area = 1000
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    print(f"검출된 윤곽선 개수: {len(contours)}")

    # 사각형 검출
    retengles = []
    for contour in contours:
        # 외곽선을 근사화하여 다각형으로 만듦
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 외곽선이 사각형인지 확인 (꼭짓점이 4개이고 닫힌 도형인지 확인)
        if len(approx) == 4:
            retengles.append(contour)

    for retengle in retengles:
        cv2.drawContours(image, [retengle], -1, (0, 255, 0), 2)

    cv2.imwrite(f"{str(time.time())[:10]}-contours.jpg", image)

    # 사각형의 좌표를 구하고 이미지를 저장
    rect_coords = []
    cropped_images = []
    for i, retengle in enumerate(retengles):
        x, y, w, h = cv2.boundingRect(retengle)
        rect_coords.append((x, y, w, h))
        cropped = gray[y:y+h, x:x+w]
        cropped_images.append(cropped)
        cv2.imwrite(f"{str(time.time())[:10]}-cropped_{i}.jpg", cropped)

    return find_most_different_image_using_matchTemplate(cropped_images, rect_coords)

def touch_rect_center(rect):
    x = rect[0] + rect[2] // 2
    y = rect[1] + rect[3] // 2
    print(f"터치 좌표: {x}, {y}")
    run_adb_command(["adb", "shell", "input", "tap", str(x), str(y)])

def main():
    print("프로그램 실행 중...")
    if not check_device_connected():
        return
    get_screen_resolution()

    print("'Space' 또는 'Enter'를 눌러서 시작하세요!")
    print("그 외 키를 누르면 종료됩니다.")

    while True:
        if keyboard.is_pressed("space") or keyboard.is_pressed("enter"):
            start_time = time.time()

            print("스크린샷 촬영 후 처리 중...")
            image_path = capture_screenshot()
            rect = get_answer_position(image_path)
            touch_rect_center(rect)

            print(f"완료! 소요시간: {time.time() - start_time:.2f} 초")

        elif keyboard.read_event().name not in ["space", "enter"]:
            print("프로그램을 종료합니다.")
            break

if __name__ == "__main__":
    main()

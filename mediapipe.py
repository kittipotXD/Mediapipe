import cv2
import mediapipe as mp
import math

# เริ่มต้นการทำงานของ MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# เปิดการใช้งานกล้อง
cap = cv2.VideoCapture(0)

# กำหนดอัตราส่วนพิกเซลต่อมิลลิเมตร (สมมุติว่า 3 พิกเซล = 1 มม.)
pixel_to_mm_ratio = 1/4

print("Press 'c' to capture an image and analyze, 'q' to quit.")

# ตัวแปรเพื่อเก็บความสูงของดวงตา
right_eye_heights = []
left_eye_heights = []
capture_count = 0  # ตัวนับจำนวนการถ่ายภาพ
max_captures = 3   # จำนวนการถ่ายภาพสูงสุด

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # แสดงภาพสด
    cv2.imshow("Webcam - Press 'c' to capture, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and capture_count < max_captures:  # ตรวจสอบจำนวนการถ่ายภาพ
        # แปลงภาพเป็น RGB เนื่องจาก MediaPipe ต้องการการแปลงนี้
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ประมวลผลเฟรมเพื่อค้นหา landmarks บนใบหน้า
        result = face_mesh.process(rgb_frame)

        # ถ้าพบ landmarks ให้ทำการดึงข้อมูลบริเวณดวงตา
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                # Index ของดวงตาขวาและซ้าย
                right_eye_top_index = 159  # ขอบบนของดวงตาขวา
                right_eye_bottom_index = 145  # ขอบล่างของดวงตาขวา
                left_eye_top_index = 386  # ขอบบนของดวงตาซ้าย
                left_eye_bottom_index = 374  # ขอบล่างของดวงตาซ้าย

                # คำนวณตำแหน่งของขอบตา
                right_eye_top = (int(face_landmarks.landmark[right_eye_top_index].x * frame.shape[1]),
                                 int(face_landmarks.landmark[right_eye_top_index].y * frame.shape[0]))
                right_eye_bottom = (int(face_landmarks.landmark[right_eye_bottom_index].x * frame.shape[1]),
                                    int(face_landmarks.landmark[right_eye_bottom_index].y * frame.shape[0]))
                left_eye_top = (int(face_landmarks.landmark[left_eye_top_index].x * frame.shape[1]),
                                int(face_landmarks.landmark[left_eye_top_index].y * frame.shape[0]))
                left_eye_bottom = (int(face_landmarks.landmark[left_eye_bottom_index].x * frame.shape[1]),
                                   int(face_landmarks.landmark[left_eye_bottom_index].y * frame.shape[0]))

                # คำนวณความสูงของตาขวาและตาซ้าย
                right_eye_height_pixels = math.dist(right_eye_top, right_eye_bottom)
                left_eye_height_pixels = math.dist(left_eye_top, left_eye_bottom)

                # แปลงความสูงจากพิกเซลเป็นมิลลิเมตร
                right_eye_height_mm = right_eye_height_pixels * pixel_to_mm_ratio
                left_eye_height_mm = left_eye_height_pixels * pixel_to_mm_ratio

                # เก็บค่าความสูงลงในลิสต์
                right_eye_heights.append(right_eye_height_mm)
                left_eye_heights.append(left_eye_height_mm)

                # วาดขอบตา
                cv2.line(frame, right_eye_top, right_eye_bottom, (0, 255, 0), 2)
                cv2.line(frame, left_eye_top, left_eye_bottom, (0, 255, 0), 2)

                # แสดงผลความสูงของดวงตา
                cv2.putText(frame, f"R Eye Height: {right_eye_height_mm:.2f} mm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"L Eye Height: {left_eye_height_mm:.2f} mm", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # แสดงค่าผลลัพธ์ในคอนโซล
                print(f"Right Eye Height: {right_eye_height_mm:.2f} mm")
                print(f"Left Eye Height: {left_eye_height_mm:.2f} mm")

                # ถ้าวัดครบ 3 รอบแล้วให้คำนวณค่าเฉลี่ย
                capture_count += 1
                if capture_count == max_captures:
                    avg_right_eye_height = sum(right_eye_heights) / len(right_eye_heights)
                    avg_left_eye_height = sum(left_eye_heights) / len(left_eye_heights)

                    print(f"Average Right Eye Height: {avg_right_eye_height:.2f} mm")
                    print(f"Average Left Eye Height: {avg_left_eye_height:.2f} mm")

                    # แสดงค่าเฉลี่ยในภาพ
                    cv2.putText(frame, f"Avg R Eye Height: {avg_right_eye_height:.2f} mm", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, f"Avg L Eye Height: {avg_left_eye_height:.2f} mm", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # ล้างข้อมูลสำหรับการวัดครั้งถัดไป
                    right_eye_heights.clear()
                    left_eye_heights.clear()

                    # ไม่อนุญาตให้ถ่ายภาพอีก
                    print("Maximum captures reached. You can no longer capture images.")
                    cv2.putText(frame, "Maximum captures reached.", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # แสดงภาพที่ถ่าย
        cv2.imshow("Captured Image", frame)

    elif key == ord('q'):
        break

# ปิดกล้องและหน้าต่างทั้งหมด
cap.release()
cv2.destroyAllWindows()

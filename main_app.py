import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import random
import os
from gtts import gTTS
from playsound import playsound

# ---------- VOICE ----------
def speak(text):
    def run():
        try:
            filename = "voice.mp3"
            gTTS(text=text, lang='en').save(filename)
            playsound(filename)
            os.remove(filename)
        except:
            pass
    threading.Thread(target=run, daemon=True).start()

# ---------- MEDIAPIPE ----------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# ---------- ANGLE ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    angle = np.abs(
        np.arctan2(c[1]-b[1], c[0]-b[0]) -
        np.arctan2(a[1]-b[1], a[0]-b[0])
    ) * 180.0 / np.pi
    return 360-angle if angle > 180 else angle

# ---------- CAMERA ----------
cap = None
for i in range(3):
    temp = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp.isOpened():
        cap = temp
        print("Camera index:", i)
        break

if cap is None:
    print("Camera not working")
    exit()

# ---------- VARIABLES ----------
stage = 1
correct_time = 0
total_time = 0
prev_time = time.time()

last_feedback = 0
last_motivation = 0
stage_announced = False

prev_angle = 0
stability_score = 0

# ---------- SETTINGS ----------
FEEDBACK_INTERVAL = 4
MOTIVATION_INTERVAL = 8

motivation_msgs = [
    "Keep going",
    "Stay steady",
    "You are doing well",
    "Maintain control",
    "Almost there"
]

stage_names = {
    1: "Isometric Bicep Hold at 90 degrees",
    2: "Controlled Elbow Flexion",
    3: "Shoulder Flexion Hold",
    4: "Shoulder Abduction Hold",
    5: "Postural Reset"
}

# ---------- LOOP ----------
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    current_time = time.time()
    delta = current_time - prev_time
    prev_time = current_time
    total_time += delta

    if not stage_announced:
        speak(stage_names[stage])
        stage_announced = True

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img)

    feedback = "Maintain posture"
    correct = False

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = result.pose_landmarks.landmark

        shoulder = [lm[12].x, lm[12].y]
        elbow = [lm[14].x, lm[14].y]
        wrist = [lm[16].x, lm[16].y]
        hip = [lm[24].x, lm[24].y]

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(hip, shoulder, elbow)

        # ---------- STAGES ----------
        if stage == 1:
            correct = 85 <= elbow_angle <= 95
            feedback = "Hold elbow at 90 degrees"

        elif stage == 2:
            correct = 40 <= elbow_angle <= 140
            feedback = "Move elbow slowly and control motion"

        elif stage == 3:
            correct = shoulder_angle > 70
            feedback = "Lift arm forward and hold steady"

        elif stage == 4:
            correct = shoulder_angle > 80
            feedback = "Lift arm sideways with control"

        elif stage == 5:
            correct = shoulder_angle < 30 and elbow_angle > 160
            feedback = "Relax and maintain posture"

        # ---------- STABILITY ----------
        angle_change = abs(elbow_angle - prev_angle)
        prev_angle = elbow_angle

        if angle_change < 5:
            stability_score += 1

        # ---------- TIMER ----------
        if correct:
            correct_time += delta
        else:
            if current_time - last_feedback > FEEDBACK_INTERVAL:
                speak(feedback)
                last_feedback = current_time

        # ---------- MOTIVATION ----------
        if correct and correct_time > 10:
            if current_time - last_motivation > MOTIVATION_INTERVAL:
                speak(random.choice(motivation_msgs))
                last_motivation = current_time

        # ---------- NEXT STAGE ----------
        if correct_time >= 120:
            stage += 1
            if stage > 5:
                stage = 1
            correct_time = 0
            stage_announced = False

    # ---------- DISPLAY ----------
    cv2.putText(frame, f"Stage: {stage}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Correct Time: {int(correct_time)} sec", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Stability: {stability_score}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.putText(frame, feedback, (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AI Rehabilitation Assistant", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------- REPORT ----------
accuracy = (correct_time / total_time) * 100 if total_time > 0 else 0

with open("physio_report.txt", "w") as f:
    f.write(f"Total Time: {int(total_time)} sec\n")
    f.write(f"Correct Time: {int(correct_time)} sec\n")
    f.write(f"Accuracy: {int(accuracy)} %\n")
    f.write(f"Stability Score: {stability_score}\n")

print("Report saved successfully!")
print("Accuracy:", int(accuracy), "%")

cap.release()
cv2.destroyAllWindows()
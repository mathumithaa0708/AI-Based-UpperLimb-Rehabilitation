import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Physio AI")
page = st.sidebar.radio("Navigation", ["🏠 Dashboard", "🏋 Exercise", "📊 Report"])

# ---------------- SESSION ----------------
def init_state():
    keys = {
        "rep": 0,
        "correct": 0,
        "angles": [],
        "rep_data": [],
        "stage": "READY",
        "feedback": "",
        "start_time": 0,
        "last_rep_time": 0,
        "angle_buffer": []
    }
    for k, v in keys.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------- ANGLE ----------------
def calc_angle(a, b, c):
    a=np.array(a); b=np.array(b); c=np.array(c)
    rad=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    ang=np.abs(rad*180/np.pi)
    if ang>180: ang=360-ang
    return ang

# ---------------- SMOOTHING ----------------
def smooth_angle(angle):
    buf = st.session_state.angle_buffer
    buf.append(angle)
    if len(buf) > 5:
        buf.pop(0)
    return np.mean(buf)

# ---------------- DASHBOARD ----------------
if page == "🏠 Dashboard":
    st.title("🏠 Dashboard")

    c1,c2,c3=st.columns(3)
    c1.metric("Total Reps", st.session_state.rep)
    c2.metric("Correct", st.session_state.correct)

    acc = 0
    if st.session_state.rep > 0:
        acc = (st.session_state.correct / st.session_state.rep) * 100

    c3.metric("Accuracy %", round(acc,2))

# ---------------- EXERCISE ----------------
elif page == "🏋 Exercise":

    st.title("🏋 Exercise Session")

    exercise = st.selectbox(
        "Select Exercise",
        ["Bicep Curl", "Arm Raise", "Shoulder Abduction"]
    )

    start = st.button("▶ Start")
    stop = st.button("⏹ Stop")

    frame_win = st.image([])

    if start:
        cap=cv2.VideoCapture(0)
        mp_pose=mp.solutions.pose
        mp_draw=mp.solutions.drawing_utils

        with mp_pose.Pose() as pose:

            while cap.isOpened():

                ret,frame=cap.read()
                if not ret: break

                img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                res=pose.process(img)
                img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

                try:
                    lm=res.pose_landmarks.landmark

                    shoulder=[lm[12].x,lm[12].y]
                    elbow=[lm[14].x,lm[14].y]
                    wrist=[lm[16].x,lm[16].y]

                    raw_angle = calc_angle(shoulder, elbow, wrist)
                    angle = smooth_angle(raw_angle)

                    st.session_state.angles.append(angle)
                    st.session_state.rep_data.append(angle)

                    now = time.time()

                    # ---------- THRESHOLDS ----------
                    STRAIGHT = 160
                    BENT = 60
                    HOLD_TIME = 0.8
                    COOLDOWN = 1.5

                    # ---------- STABLE STATES ----------
                    if angle > STRAIGHT:
                        if st.session_state.stage == "DOWN":
                            # rep completed
                            if now - st.session_state.last_rep_time > COOLDOWN:

                                st.session_state.rep += 1
                                st.session_state.last_rep_time = now

                                # ANALYSIS
                                data = st.session_state.rep_data
                                rom = max(data) - min(data)
                                stability = np.std(data)
                                duration = now - st.session_state.start_time

                                if rom > 90 and stability < 15 and 1 < duration < 4:
                                    st.session_state.correct += 1
                                    st.session_state.feedback = "✅ Perfect"
                                else:
                                    st.session_state.feedback = "⚠️ Improve"

                                st.session_state.rep_data = []

                            st.session_state.stage = "READY"

                        else:
                            st.session_state.stage = "READY"

                        instruction = "Hold Straight"

                    elif angle < BENT:
                        if st.session_state.stage == "READY":
                            # valid movement start
                            st.session_state.stage = "DOWN"
                            st.session_state.start_time = now

                        instruction = "Bend Arm"

                    else:
                        instruction = "Moving..."

                    # ---------- DISPLAY ----------
                    cv2.putText(img,f'{exercise}',(10,30),0,1,(255,255,255),2)
                    cv2.putText(img,f'Angle:{int(angle)}',(10,60),0,1,(0,255,0),2)
                    cv2.putText(img,f'Reps:{st.session_state.rep}',(10,90),0,1,(255,0,0),2)
                    cv2.putText(img,f'Stage:{st.session_state.stage}',(10,120),0,1,(0,255,255),2)
                    cv2.putText(img,instruction,(10,150),0,1,(0,0,255),2)
                    cv2.putText(img,st.session_state.feedback,(10,180),0,1,(255,255,0),2)

                except:
                    pass

                mp_draw.draw_landmarks(img,res.pose_landmarks,mp_pose.POSE_CONNECTIONS)
                frame_win.image(img)

                if stop:
                    break

        cap.release()

# ---------------- REPORT ----------------
elif page == "📊 Report":

    st.title("📊 Report")

    total = st.session_state.rep
    correct = st.session_state.correct

    acc = 0
    if total > 0:
        acc = (correct / total) * 100

    st.metric("Total Reps", total)
    st.metric("Correct Reps", correct)
    st.metric("Accuracy", round(acc,2))

    if len(st.session_state.angles) > 0:
        fig, ax = plt.subplots()
        ax.plot(st.session_state.angles)
        ax.set_title("Angle Graph")
        st.pyplot(fig)
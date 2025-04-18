import pygame
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mpHands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pygame.init()
pygame.mixer.init()

cap = cv2.VideoCapture(0)

# Finger tip and joint ids
tip_ids = [4, 8, 12, 16, 20]
base_ids = [3, 6, 10, 14, 18]

c4=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448539__tedagame__c4.ogg")
d4=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448602__tedagame__d4.ogg")
e4=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448613__tedagame__e4.ogg")
f4=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448585__tedagame__f4.ogg")
g4=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448592__tedagame__g4.ogg")
a4=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448577__tedagame__a4.ogg")
b3=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448568__tedagame__b3.ogg")
a5=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448576__tedagame__a5.ogg")
b2=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448569__tedagame__b2.ogg")
c5=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448548__tedagame__c5.ogg")
d5=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448603__tedagame__d5.ogg")
e5=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448612__tedagame__e5.ogg")
f5=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448594__tedagame__f5.ogg")
g5=pygame.mixer.Sound(r"C:\Users\hp\PycharmProjects\VirtualPiano\25405__tedagame__88-piano-keys-long-reverb\448599__tedagame__g5.ogg")


notes=[c4,d4,e4,f4,g4,a4,b3,a5,b2,c5,d5,e5,f5,g5]

# Prevent repeat triggering
finger_states = {}  # (hand_label, finger_id): bool

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mpHands.process(img_rgb)

    # Draw 14 vertical key lines
    for i in range(1, 14):
        x_pos = i * 45
        cv2.line(img, (x_pos, 0), (x_pos, h), (255, 255, 255), 2)
        cv2.putText(img, str(i), (x_pos - 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # "Left" or "Right"

            for tip_id, base_id in zip(tip_ids, base_ids):
                tip = hand_landmarks.landmark[tip_id]
                base = hand_landmarks.landmark[base_id]

                x = int(tip.x * w)
                y = int(tip.y * h)

                key_index = min(x // 45, 13)
                is_pressed = tip.y < base.y

                finger_id = (hand_label, tip_id)
                if is_pressed and not finger_states.get(finger_id, False):
                    notes[key_index].play()
                    finger_states[finger_id] = True
                    cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                elif not is_pressed:
                    finger_states[finger_id] = False

    cv2.imshow("Virtual Piano", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

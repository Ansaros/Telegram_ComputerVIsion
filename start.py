from ultralytics import YOLO
import cv2
import telebot  

bot = telebot.TeleBot("TOKEN")  
chat_id = -1002064116483 

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("C:/Users/Пользователь/Desktop/Work/telegacam/best.pt")

classNames = ["0"]

#confidence threshold
confidence_threshold = 0.5 

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = box.conf[0]
            if confidence >= confidence_threshold:
                #Bound box(Need to end)
                ...

                cv2.imwrite("screenshot.jpg", img) 
                with open("screenshot.jpg", "rb") as img_file:
                    bot.send_photo(chat_id, img_file) 

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
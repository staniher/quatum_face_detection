import cv2
import numpy as np
import requests
import tensorflow as tf
import time

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    return interpreter, input_index, output_index

def preprocess_face(face_img, target_size=(160, 160)):
    face_img = cv2.resize(face_img, target_size)
    face_img = face_img / 255.0
    return np.expand_dims(face_img.astype(np.float32), axis=0)

def extract_embedding(interpreter, input_index, output_index, face_img):
    interpreter.set_tensor(input_index, face_img)
    interpreter.invoke()
    return interpreter.get_tensor(output_index)[0]

def main():
    model_path = "face_embedder.tflite"
    server_url = "http://<IP_DU_SERVEUR>:5000/predict"
    interpreter, input_index, output_index = load_tflite_model(model_path)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            preprocessed = preprocess_face(face)
            embedding = extract_embedding(interpreter, input_index, output_index, preprocessed)
            try:
                response = requests.post(server_url, json={"vector": embedding.tolist()}, timeout=5)
                label = response.json().get("label", "inconnu")
            except:
                label = "serveur non dispo"

            color = (0, 255, 0) if label == "autorise" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Controle d'acces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

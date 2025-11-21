import cv2

url = "http://192.168.0.50:8080/video"  # IP de tu tablet
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("❌ No se pudo conectar con la cámara de la tablet.")
    exit()

print("✅ Conectado. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    cv2.imshow("Camara Tablet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

# URL RTSP del canal 4 de tu grabador Dahua
url = "rtsp://po:+Admin10@10.6.31.57:554/cam/realmonitor?channel=4&subtype=0"

# Intentar abrir con ffmpeg (mejor compatibilidad en Windows/Linux)
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("❌ No se pudo abrir el stream. Revisa URL o red.")
    exit()

print("✅ Conectado al grabador. Mostrando canal 4...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se recibió frame. Puede ser pérdida de conexión.")
        break

    cv2.imshow("Camara Canal 4 (Grabador)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

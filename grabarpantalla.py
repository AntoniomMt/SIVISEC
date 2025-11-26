import cv2
import numpy as np
import pyautogui
import time

# Nombre del archivo de salida
filename = "grabacion.mp4"

# Espera antes de empezar (por ejemplo, 3 segundos)
print("Preparando grabación...")
time.sleep(3)

# Obtener resolución de pantalla
screen_size = pyautogui.size()

# Crear objeto para escribir el video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(filename, fourcc, 20.0, screen_size)

print("Grabando pantalla...")
print("Presiona 'q' en la ventana para detener la grabación.")

while True:
    # Capturar pantalla con pyautogui
    img = pyautogui.screenshot()
    frame = np.array(img)
    # Convertir de RGB a BGR (OpenCV usa BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Mostrar vista previa
    cv2.imshow("Grabando...", frame)

    # Guardar frame
    out.write(frame)

    # Parar con tecla q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Grabación detenida por usuario.")
        break

# Liberar recursos
out.release()
cv2.destroyAllWindows()

print(f"✅ Video guardado como {filename}")

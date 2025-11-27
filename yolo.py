from ultralytics import YOLO
import cv2

# Cambia la ruta a tu modelo entrenado
RUTA = "runs/train_custom/exp_20251127_003246/weights/best.pt"

def main():
    print("Cargando modelo:", RUTA)
    modelo = YOLO(RUTA)

    cap = cv2.VideoCapture(0)  # Webcam

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return

    print("Listo. Presiona ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error leyendo frame")
            break

        # Inferencia
        results = modelo(frame, verbose=False)

        # Dibujar boxes del modelo entrenado
        annotated_frame = results[0].plot()

        # Mostrar ventana
        cv2.imshow("Detector de personas y bolsas", annotated_frame)

        # ESC para salir
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

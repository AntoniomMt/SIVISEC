import tkinter as tk
import threading
import cv2
from PIL import Image, ImageTk
import random
import datetime
import time
import numpy as np
from tkinter import messagebox, ttk
#from base import GestorBaseDatos
import mediapipe as mp
from ultralytics import YOLO

# ===============================
#   MODELO
# ===============================
class SistemaSeguridad:
    def __init__(self):
        self.estado_sistema = "Detenido"
        self.umbral_confianza = 70
        self.camaras_activas = []

    def agregar_camara(self, id_camara, nombre, ubicacion):
        self.camaras_activas.append({
            "id": id_camara,
            "nombre": nombre,
            "ubicacion": ubicacion
        })

    def cambiar_umbral(self, nuevo_umbral):
        if 50 <= nuevo_umbral <= 100:
            self.umbral_confianza = nuevo_umbral
            return True
        return False


# ===============================
#   VISTA (Tkinter)
# ===============================
class SistemaSeguridadVista:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Seguridad con Detección de Botellas")
        self.root.geometry("1000x700")
        self.video_captures = {}

        # Frame superior
        frame_top = tk.Frame(root, bg="#20232a")
        frame_top.pack(fill="x")

        self.lbl_estado = tk.Label(frame_top, text="Estado: Detenido", bg="#20232a", fg="white", font=("Arial", 14))
        self.lbl_estado.pack(side="left", padx=10, pady=10)

        tk.Label(frame_top, text="Umbral detección:", bg="#20232a", fg="white").pack(side="left", padx=10)
        self.threshold_var = tk.IntVar(value=70)
        self.scale = ttk.Scale(frame_top, from_=50, to=100, orient="horizontal", variable=self.threshold_var)
        self.scale.pack(side="left", padx=5)

        self.btn_iniciar = ttk.Button(frame_top, text="Iniciar Monitoreo")
        self.btn_iniciar.pack(side="left", padx=10)
        self.btn_detener = ttk.Button(frame_top, text="Detener")
        self.btn_detener.pack(side="left", padx=10)
        self.btn_agregar_camara = ttk.Button(frame_top, text="Agregar Cámara")
        self.btn_agregar_camara.pack(side="left", padx=10)

        # Frame de video
        self.frame_videos = tk.Frame(root, bg="#282c34")
        self.frame_videos.pack(fill="both", expand=True, padx=10, pady=10)

    def crear_feed_video(self, id_camara, nombre):
        frame = tk.Frame(self.frame_videos, bg="#333333", bd=2, relief="groove")
        frame.pack(side="left", padx=10, pady=10)
        tk.Label(frame, text=nombre, fg="white", bg="#333333").pack()
        label = tk.Label(frame, bg="black")
        label.pack(padx=5, pady=5)
        self.video_captures[id_camara] = label

    def actualizar_estado(self, texto):
        self.lbl_estado.config(text=f"Estado: {texto}")

    def mostrar_mensaje(self, titulo, mensaje):
        messagebox.showinfo(titulo, mensaje)


# ===============================
#   DETECTOR (YOLO + Mediapipe)
# ===============================
class DetectorBotellaMano:
    def __init__(self, modelo_path="yolov5nu.pt"):
        self.model_yolo = YOLO(modelo_path)
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[DEBUG] Clases YOLO:", self.model_yolo.names)

    def procesar_frame(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.holistic.process(rgb)

            # Dibuja manos
            if result.left_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, result.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if result.right_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, result.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

            # Detección de botellas
            results = self.model_yolo(frame, verbose=False)
            botella_boxes = []
            for r in results:
                for box in r.boxes:
                    cls_name = self.model_yolo.names[int(box.cls[0])]
                    conf = float(box.conf[0])
                    if cls_name.lower() == "bottle" and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        botella_boxes.append((x1, y1, x2, y2))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Botella ({conf:.2f})", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Detectar si mano toca botella
            def mano_toca_botella(landmarks, boxes, w, h):
                if landmarks:
                    for lm in landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        for (x1, y1, x2, y2) in boxes:
                            if x1 - 10 < x < x2 + 10 and y1 - 10 < y < y2 + 10:
                                return True
                return False

            h, w, _ = frame.shape
            toca = (
                mano_toca_botella(result.left_hand_landmarks, botella_boxes, w, h) or
                mano_toca_botella(result.right_hand_landmarks, botella_boxes, w, h)
            )

            if toca:
                cv2.putText(frame, "Mano cerca de botella", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                print("[Detector] Mano cerca de botella detectada")

        except Exception as e:
            print("[Detector] Error:", e)

        return frame


# ===============================
#   CONTROLADOR
# ===============================
class SistemaSeguridadControlador:
    def __init__(self, root):
        self.root = root
        self.modelo = SistemaSeguridad()
        self.vista = SistemaSeguridadVista(root)
        self.detector = DetectorBotellaMano("yolov5nu.pt")

        self.monitoreo_activo = False
        self.hilos_camara = {}

        # Vincular botones
        self.vista.btn_iniciar.config(command=self.iniciar_monitoreo)
        self.vista.btn_detener.config(command=self.detener_monitoreo)
        self.vista.btn_agregar_camara.config(command=self.agregar_camara)

        self.agregar_camara_real("CAM1", "Cámara Principal", "Webcam")

    def agregar_camara_real(self, id_camara, nombre, ubicacion):
        self.modelo.agregar_camara(id_camara, nombre, ubicacion)
        self.vista.crear_feed_video(id_camara, f"{nombre} ({ubicacion})")

    def iniciar_monitoreo(self):
        if self.monitoreo_activo:
            return
        self.monitoreo_activo = True
        self.vista.actualizar_estado("Monitoreo")
        for cam in self.modelo.camaras_activas:
            hilo = threading.Thread(target=self.procesar_video, args=(cam["id"],))
            hilo.daemon = True
            hilo.start()
            self.hilos_camara[cam["id"]] = hilo
        self.vista.mostrar_mensaje("Monitoreo", "Monitoreo iniciado con cámara real")

    def detener_monitoreo(self):
        self.monitoreo_activo = False
        self.vista.actualizar_estado("Detenido")
        self.vista.mostrar_mensaje("Monitoreo", "Monitoreo detenido")

    def procesar_video(self, id_camara):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[Error] No se pudo acceder a la cámara.")
            return

        while self.monitoreo_activo:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            frame = self.detector.procesar_frame(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)

            label = self.vista.video_captures.get(id_camara)
            if label:
                lbl_w = label.winfo_width()
                lbl_h = label.winfo_height()
                if lbl_w > 1 and lbl_h > 1:
                    img = img.resize((lbl_w, lbl_h))
                img_tk = ImageTk.PhotoImage(img)
                self.root.after(0, lambda: label.config(image=img_tk))
                label.imgtk = img_tk

            time.sleep(0.03)

        cap.release()


# ===============================
#   EJECUCIÓN PRINCIPAL
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaSeguridadControlador(root)
    root.mainloop()
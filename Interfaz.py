import tkinter as tk
import threading
import cv2
import random
import datetime
import time
import numpy as np
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
#from base import GestorBaseDatos

# INICIO DE LA CLASE SistemaSeguridad (antes modelo.py)
class SistemaSeguridad:
    def __init__(self):
        """Inicializa el sistema de seguridad con valores predeterminados."""
        self.historial_alertas = []  # Lista de alertas generadas
        self.estado_sistema = "Monitoreo"  # Estado del sistema (Monitoreo, Alerta, etc.)
        self.camaras_activas = []  # Lista de cámaras activas en el sistema
        self.umbral_confianza = 80  # Umbral mínimo de confianza para alertas (en %)

    def cambiar_umbral(self, nuevo_umbral):
        if 50 <= nuevo_umbral <= 100:
            self.umbral_confianza = nuevo_umbral
            return True
        return False

    def agregar_camara(self, id_camara, nombre, ubicacion):
        self.camaras_activas.append({"id": id_camara, "nombre": nombre, "ubicacion": ubicacion})

    def marcar_alerta_revisada(self, id_alerta):
        for alerta in self.historial_alertas:
            if alerta["id_alerta"] == id_alerta:
                alerta["estado"] = "Revisada"
                return True
        return False


# INICIO DE LA CLASE SistemaSeguridadVista (antes vista.py)
class SistemaSeguridadVista:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Seguridad SIVISEC")
        self.root.state('zoomed')
        self.root.configure(bg="#f0f0f0")

        # Variables
        self.threshold_var = tk.IntVar(value=70)
        self.video_captures = {}

        # Título
        self.titulo_principal = tk.Label(
            root, text="Sistema de Seguridad SIVISEC", font=("Arial", 16, "bold"), bg="#f0f0f0"
        )
        self.titulo_principal.pack(pady=10, anchor="w", padx=20)

        # Estado
        self.frame_estado = tk.Frame(root, bg="#f0f0f0")
        self.frame_estado.place(relx=0.99, rely=0.06, anchor="e")
        self.label_estado = tk.Label(self.frame_estado, text="Estado: En mantenimiento", font=("Arial", 12, "bold"), fg="red", bg="#f0f0f0")
        self.label_estado.pack()

        # Controles
        self.frame_controles = tk.Frame(root, bg="#f0f0f0")
        self.frame_controles.pack(fill="x", padx=20, pady=5)

        self.btn_iniciar = tk.Button(self.frame_controles, text="Iniciar Monitoreo", width=15)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        self.btn_detener = tk.Button(self.frame_controles, text="Detener", width=10)
        self.btn_detener.pack(side=tk.LEFT, padx=5)

        tk.Label(self.frame_controles, text="Umbral de alerta (%):", bg="#f0f0f0").pack(side=tk.LEFT, padx=(20, 0))
        self.slider_umbral = tk.Scale(self.frame_controles, variable=self.threshold_var, from_=50, to=100, orient="horizontal", length=150, bg="#f0f0f0")
        self.slider_umbral.pack(side=tk.LEFT)
        tk.Label(self.frame_controles, textvariable=self.threshold_var, bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        self.btn_aplicar = tk.Button(self.frame_controles, text="Aplicar", width=10)
        self.btn_aplicar.pack(side=tk.LEFT, padx=5)

        # Columnas principales
        self.frame_principal = tk.Frame(root, bg="#f0f0f0")
        self.frame_principal.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Cámaras
        self.frame_camaras = tk.LabelFrame(self.frame_principal, text="Monitoreo", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.frame_camaras.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        self.frame_grid_camaras = tk.Frame(self.frame_camaras, bg="#f0f0f0")
        self.frame_grid_camaras.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Alertas
        self.frame_alertas = tk.LabelFrame(self.frame_principal, text="Alertas y eventos", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.frame_alertas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        tk.Label(self.frame_alertas, text="Alertas activas", bg="#f0f0f0", anchor="w").pack(fill="x", pady=(5, 0))
        
        # Frame contenedor para las alertas activas y su scrollbar
        alertas_container = tk.Frame(self.frame_alertas)
        alertas_container.pack(fill="x", pady=5)
        
        # Crear canvas y scrollbar
        alertas_canvas = tk.Canvas(alertas_container, height=120, bg="#ffcccc")
        alertas_scrollbar = ttk.Scrollbar(alertas_container, orient="vertical", command=alertas_canvas.yview)
        
        # Frame interior para el contenido
        self.frame_alertas_activas = tk.Frame(alertas_canvas, bg="#ffcccc")
        
        # Configurar el canvas
        alertas_canvas.configure(yscrollcommand=alertas_scrollbar.set)
        
        # Empaquetar los elementos
        alertas_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        alertas_canvas.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Crear ventana en el canvas con el frame
        canvas_frame = alertas_canvas.create_window((0, 0), window=self.frame_alertas_activas, anchor="nw", width=alertas_canvas.winfo_width())
        
        # Configurar eventos para actualizar el scrolling
        def on_frame_configure(event):
            alertas_canvas.configure(scrollregion=alertas_canvas.bbox("all"))
        
        def on_canvas_configure(event):
            alertas_canvas.itemconfig(canvas_frame, width=event.width)
        
        self.frame_alertas_activas.bind("<Configure>", on_frame_configure)
        alertas_canvas.bind("<Configure>", on_canvas_configure)

        tk.Label(self.frame_alertas, text="Historial de Eventos", bg="#f0f0f0", anchor="w").pack(fill="x", pady=(10, 0))
        
        # Frame contenedor para el Treeview y scrollbars
        tree_frame = tk.Frame(self.frame_alertas)
        tree_frame.pack(fill="both", expand=True, pady=5)
        
        # Crear scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configurar Treeview con scrollbars
        columns = ("ID", "Fecha/Hora", "Cámara", "Tipo")
        self.history_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10,
                                       yscrollcommand=tree_scroll_y.set,
                                       xscrollcommand=tree_scroll_x.set)
        
        # Configurar scrollbars
        tree_scroll_y.config(command=self.history_tree.yview)
        tree_scroll_x.config(command=self.history_tree.xview)
        
        # Configurar columnas
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)
        
        self.history_tree.pack(side=tk.LEFT, fill="both", expand=True)

        # Botón agregar cámara
        self.btn_agregar_camara = tk.Button(self.frame_camaras, text="Agregar cámara")
        self.btn_agregar_camara.pack(side=tk.BOTTOM, pady=7)

        # Barra inferior
        self.frame_barra_estado = tk.Frame(root, bg="#d9d9d9", height=25)
        self.frame_barra_estado.pack(fill="x", side=tk.BOTTOM)
        self.label_fecha_hora = tk.Label(
            self.frame_barra_estado,
            text=f"Fecha y hora: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            bg="#d9d9d9"
        )
        self.label_fecha_hora.pack(side=tk.LEFT, padx=10)
        self.actualizar_fecha_hora()

    def actualizar_fecha_hora(self):
        self.label_fecha_hora.config(
            text=f"Fecha y hora: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        )
        self.root.after(1000, self.actualizar_fecha_hora)

    def actualizar_estado(self, estado):
        self.label_estado.config(
            text=f"Estado: {estado}",
            fg="red" if estado == "Alerta" else "green"
        )

    def mostrar_mensaje(self, titulo, mensaje):
        messagebox.showinfo(titulo, mensaje)

    def crear_feed_video(self, id_camara, nombre):
        num_camaras = len(self.video_captures)
        frame = tk.Frame(self.frame_grid_camaras, borderwidth=2, relief="groove", bg="white")
        frame.grid(row=num_camaras, column=0, padx=10, pady=10, sticky="nsew")
        label_nombre = tk.Label(frame, text=nombre, bg="white", font=("Arial", 10, "bold"))
        label_nombre.pack(anchor="w", padx=5, pady=5)
        video_panel = tk.Label(frame, bg="black")
        video_panel.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.video_captures[id_camara] = video_panel
        self.frame_grid_camaras.grid_columnconfigure(0, weight=1)
        self.frame_grid_camaras.grid_rowconfigure(num_camaras, weight=1)


# CONTROLADOR
class SistemaSeguridadControlador:
    def __init__(self, root):
        self.root = root
        self.modelo = SistemaSeguridad()
        self.vista = SistemaSeguridadVista(root)
        #self.gestor_bd = GestorBaseDatos()
        self.monitoreo_activo = False
        self.hilos_camara = {}
        self.capturas_video = {}
        self.alertas_programadas = []

        # Conexión vista-controlador
        self.vista.btn_iniciar.config(command=self.iniciar_monitoreo)
        self.vista.btn_detener.config(command=self.detener_monitoreo)
        self.vista.btn_aplicar.config(command=self.aplicar_umbral)
        self.vista.btn_agregar_camara.config(command=self.agregar_camara)

        # Agregar cámaras reales
        self.agregar_camara_real("0", "Webcam", "Frente")
        self.agregar_camara_real("1", "Webcam", "Aux")

    def agregar_camara_real(self, id_camara, nombre, ubicacion):
        if any(cam["id"] == id_camara for cam in self.modelo.camaras_activas):
            return
        try:
            cap = cv2.VideoCapture(int(id_camara))
            if not cap.isOpened():
                self.vista.mostrar_mensaje("Error", f"No se pudo abrir cámara {id_camara}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capturas_video[id_camara] = cap
            self.modelo.agregar_camara(id_camara, nombre, ubicacion)
            self.vista.crear_feed_video(id_camara, f"{nombre} ({ubicacion})")
        except Exception as e:
            self.vista.mostrar_mensaje("Error", str(e))

    # ✅ CORREGIDO
    def iniciar_monitoreo(self):
        """Inicia o reinicia correctamente las cámaras"""
        if self.monitoreo_activo:
            return
        self.monitoreo_activo = True
        self.modelo.estado_sistema = "Monitoreo"
        self.vista.actualizar_estado("Monitoreo")

        # Reabrir cámaras si están cerradas
        for camara in self.modelo.camaras_activas:
            id_camara = camara["id"]
            if id_camara.isdigit():
                if id_camara not in self.capturas_video or not self.capturas_video[id_camara].isOpened():
                    cap = cv2.VideoCapture(int(id_camara))
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.capturas_video[id_camara] = cap
                    else:
                        self.vista.mostrar_mensaje("Error", f"No se pudo reabrir cámara {id_camara}")
                        continue
            hilo = threading.Thread(target=self.procesar_video, args=(id_camara,))
            hilo.daemon = True
            hilo.start()
            self.hilos_camara[id_camara] = hilo

        self.vista.mostrar_mensaje("Monitoreo", "El monitoreo se ha iniciado")

    def detener_monitoreo(self):
        if not self.monitoreo_activo:
            return
        self.monitoreo_activo = False
        self.modelo.estado_sistema = "Detenido"
        self.vista.actualizar_estado("Detenido")
        for cap in self.capturas_video.values():
            if cap.isOpened():
                cap.release()
        self.vista.mostrar_mensaje("Monitoreo", "El monitoreo se ha detenido")

    def procesar_video(self, id_camara):
        from Bueno import detectar_personas  # Llamar a tu módulo aquí
        while self.monitoreo_activo:
            if id_camara not in self.capturas_video:
                break
            cap = self.capturas_video[id_camara]
            ret, frame = cap.read()
            if not ret:
                break

            # --- Aplicar tu programa de detección aquí ---
            frame, _ = detectar_personas(frame)  # Ignoramos alertas, solo modificamos el frame

            # Convertir a RGB para tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb).resize((320, 240))
            img_tk = ImageTk.PhotoImage(image=img)
            self.vista.root.after(0, self.actualizar_label_video, id_camara, img_tk)

            time.sleep(0.03)

    def actualizar_label_video(self, id_camara, img_tk):
        if id_camara in self.vista.video_captures:
            label = self.vista.video_captures[id_camara]
            label.imgtk = img_tk
            label.configure(image=img_tk)

    def aplicar_umbral(self):
        nuevo = self.vista.threshold_var.get()
        if self.modelo.cambiar_umbral(nuevo):
            self.vista.mostrar_mensaje("Umbral", f"Nuevo umbral: {nuevo}%")
        else:
            self.vista.mostrar_mensaje("Error", "El umbral debe estar entre 50 y 100")

    def agregar_camara(self):
        self.vista.mostrar_mensaje("Aviso", "Función de agregar cámara pendiente")

    def __del__(self):
        for cap in self.capturas_video.values():
            if cap.isOpened():
                cap.release()


# MAIN
if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaSeguridadControlador(root)
    root.mainloop()

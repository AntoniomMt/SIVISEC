import tkinter as tk
import threading
import cv2
import random
import datetime
import time
import numpy as np
from PIL import Image, ImageTk
from tkinter import messagebox, ttk

# INICIO DE LA CLASE SistemaSeguridad (antes modelo.py)
class SistemaSeguridad:
    def __init__(self):
        """Inicializa el sistema de seguridad con valores predeterminados."""
        self.historial_alertas = []
        self.estado_sistema = "Monitoreo"
        self.camaras_activas = []
        self.contador_alertas = 0

    def agregar_camara(self, id_camara, nombre, ubicacion):
        self.camaras_activas.append({"id": id_camara, "nombre": nombre, "ubicacion": ubicacion})

    def eliminar_camara(self, id_camara):
        self.camaras_activas = [cam for cam in self.camaras_activas if cam["id"] != id_camara]

    def agregar_alerta(self, id_camara, tipo_alerta, confianza=0):
        """Agrega una nueva alerta al sistema"""
        self.contador_alertas += 1
        alerta = {
            "id_alerta": self.contador_alertas,
            "fecha_hora": datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            "id_camara": id_camara,
            "tipo": tipo_alerta,
            "confianza": confianza,
            "estado": "Activa"
        }
        self.historial_alertas.append(alerta)
        return alerta

    def marcar_alerta_revisada(self, id_alerta):
        for alerta in self.historial_alertas:
            if alerta["id_alerta"] == id_alerta:
                alerta["estado"] = "Revisada"
                return True
        return False

    def obtener_alertas_filtradas(self, filtro="Todas"):
        """Obtiene alertas según el filtro seleccionado"""
        if filtro == "Todas":
            return self.historial_alertas
        else:
            return [a for a in self.historial_alertas if a["tipo"] == filtro]

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
        self.alertas_activas_widgets = []

        # Título
        self.titulo_principal = tk.Label(
            root, text="Sistema de Seguridad SIVISEC", font=("Arial", 16, "bold"), bg="#f0f0f0"
        )
        self.titulo_principal.pack(pady=10, anchor="w", padx=20)

        # Estado
        self.frame_estado = tk.Frame(root, bg="#f0f0f0")
        self.frame_estado.place(relx=0.99, rely=0.06, anchor="e")
        self.label_estado = tk.Label(self.frame_estado, text="Estado: En mantenimiento", 
                                     font=("Arial", 12, "bold"), fg="red", bg="#f0f0f0")
        self.label_estado.pack()

        # Controles
        self.frame_controles = tk.Frame(root, bg="#f0f0f0")
        self.frame_controles.pack(fill="x", padx=20, pady=5)

        self.btn_iniciar = tk.Button(self.frame_controles, text="Iniciar Monitoreo", width=15)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        self.btn_detener = tk.Button(self.frame_controles, text="Detener", width=10)
        self.btn_detener.pack(side=tk.LEFT, padx=5)

        # Separador visual
        separador1 = tk.Frame(self.frame_controles, width=2, bg="#cccccc")
        separador1.pack(side=tk.LEFT, fill="y", padx=15, pady=5)

        # Filtro de alertas - TOTALMENTE HORIZONTAL SIN RECUADRO
        tk.Label(self.frame_controles, text="🔍", bg="#f0f0f0", 
                font=("Arial", 12)).pack(side=tk.LEFT, padx=(0, 5))

        # Canvas para el slider - MÁS ANCHO PARA 5 OPCIONES
        self.slider_canvas = tk.Canvas(self.frame_controles, width=500, height=35, 
                                       bg="#f0f0f0", highlightthickness=0, bd=0)
        self.slider_canvas.pack(side=tk.LEFT, padx=5)

        # Variables para el slider - CORREGIDO: 5 OPCIONES
        self.filtros_opciones = ["Todas", "Sosteniendo mercancía", "Comportamiento sospechoso", "Escondiendo mercancía", "Posible robo"]
        self.filtro_index = 0
        self.slider_arrastrando = False

        # Colores para cada filtro
        self.colores_filtro = {
            "Todas": "#26E439",
            "Sosteniendo mercancía": "#ECDA38",
            "Comportamiento sospechoso": "#D777A1",
            "Escondiendo mercancía": "#FF9800",
            "Posible robo": "#CB1616"
        }

        self.filtro_var = tk.StringVar(value="Todas")

        # Badge del filtro actual - INLINE
        self.label_filtro_actual = tk.Label(self.frame_controles, text="Todas", 
                                           bg="#26E439", fg="white",
                                           font=("Arial", 8, "bold"),
                                           relief="flat", bd=0,
                                           padx=8, pady=4)
        self.label_filtro_actual.pack(side=tk.LEFT, padx=8)

        # Dibujar el slider
        self.root.after(100, self.dibujar_slider)

        # Eventos del mouse
        self.slider_canvas.bind("<Button-1>", self.slider_click)
        self.slider_canvas.bind("<B1-Motion>", self.slider_arrastrar)
        self.slider_canvas.bind("<ButtonRelease-1>", self.slider_soltar)

        # ==================== PESTAÑAS ====================
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Estilo para las pestañas
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=('Arial', 11, 'bold'), padding=[20, 10])

        # ==================== PESTAÑA 1: VISTA COMPLETA ====================
        self.tab_completa = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_completa, text="Vista completa")

        # Columnas principales en vista completa
        frame_principal_completa = tk.Frame(self.tab_completa, bg="#f0f0f0")
        frame_principal_completa.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Cámaras en vista completa
        self.frame_camaras_completa = tk.LabelFrame(frame_principal_completa, text="Monitoreo", 
                                          font=("Arial", 12, "bold"), bg="#f0f0f0", width=1000)
        self.frame_camaras_completa.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        self.frame_camaras_completa.pack_propagate(False)
        
        self.frame_grid_camaras_completa = tk.Frame(self.frame_camaras_completa, bg="#f0f0f0")
        self.frame_grid_camaras_completa.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_grid_camaras_completa.grid_rowconfigure(0, weight=1)
        self.frame_grid_camaras_completa.grid_rowconfigure(1, weight=1)
        self.frame_grid_camaras_completa.grid_columnconfigure(0, weight=1)
        self.frame_grid_camaras_completa.grid_columnconfigure(1, weight=1)

        # Alertas en vista completa
        self.frame_alertas_completa = tk.LabelFrame(frame_principal_completa, text="Alertas y eventos", 
                                          font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.frame_alertas_completa.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self._crear_panel_alertas(self.frame_alertas_completa, "completa")

        # Botón agregar cámara en vista completa
        self.btn_agregar_camara_completa = tk.Button(self.frame_camaras_completa, text="Agregar cámara")
        self.btn_agregar_camara_completa.pack(side=tk.BOTTOM, pady=7)

        # ==================== PESTAÑA 2: SOLO CÁMARAS ====================
        self.tab_camaras = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_camaras, text="Cámaras")

        self.frame_camaras_solo = tk.LabelFrame(self.tab_camaras, text="Monitoreo de Cámaras", 
                                          font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.frame_camaras_solo.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.frame_grid_camaras_solo = tk.Frame(self.frame_camaras_solo, bg="#f0f0f0")
        self.frame_grid_camaras_solo.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_grid_camaras_solo.grid_rowconfigure(0, weight=1)
        self.frame_grid_camaras_solo.grid_rowconfigure(1, weight=1)
        self.frame_grid_camaras_solo.grid_columnconfigure(0, weight=1)
        self.frame_grid_camaras_solo.grid_columnconfigure(1, weight=1)

        # Botón agregar cámara en vista solo cámaras
        self.btn_agregar_camara_solo = tk.Button(self.frame_camaras_solo, text="Agregar cámara")
        self.btn_agregar_camara_solo.pack(side=tk.BOTTOM, pady=7)

        # ==================== PESTAÑA 3: SOLO ALERTAS ====================
        self.tab_alertas = tk.Frame(self.notebook, bg="#f0f0f0")
        self.notebook.add(self.tab_alertas, text="Alertas / Historial")

        self.frame_alertas_solo = tk.LabelFrame(self.tab_alertas, text="Gestión de Alertas y Eventos", 
                                          font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.frame_alertas_solo.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self._crear_panel_alertas(self.frame_alertas_solo, "solo")

        # ==================== REFERENCIAS ====================
        # Usaremos el grid de la vista completa como principal
        self.frame_grid_camaras = self.frame_grid_camaras_completa
        self.btn_agregar_camara = self.btn_agregar_camara_completa

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

    def dibujar_slider(self):
        """Dibuja el slider personalizado con 5 OPCIONES"""
        self.slider_canvas.delete("all")
        
        # Colores para cada sección
        colores = ["#26E439", "#ECDA38", "#D777A1", "#FF9800", "#CB1616"]
        labels = ["Todas", "Tipo 1", "Tipo 2", "Tipo 3", "Tipo 4"]
        
        # Dimensiones - CORREGIDO PARA 5 OPCIONES
        ancho_total = 480
        alto = 40
        inicio_x = 20
        ancho_seccion = ancho_total / 5  # DIVIDIR ENTRE 5
        
        # Dibujar línea de fondo (track)
        self.slider_canvas.create_rectangle(inicio_x, 18, inicio_x + ancho_total, 22,
                                           fill="#e0e0e0", outline="#cccccc", width=1)
        
        # Dibujar marcadores de posición - 5 MARCADORES
        for i in range(5):
            x = inicio_x + (i * ancho_seccion) + (ancho_seccion / 2)
            
            # Círculo de marcador
            radio = 4 if i == self.filtro_index else 3
            color = colores[i] if i == self.filtro_index else "#bdbdbd"
            
            self.slider_canvas.create_oval(x - radio, 20 - radio, x + radio, 20 + radio,
                                          fill=color, outline="#999999", width=1)
            
            # Etiquetas ARRIBA
            font_weight = "bold" if i == self.filtro_index else "normal"
            color_texto = colores[i] if i == self.filtro_index else "#757575"
            
            font_size = 7 if i == self.filtro_index else 7
            
            self.slider_canvas.create_text(x, 8, text=labels[i], 
                                          font=("Arial", font_size, font_weight),
                                          fill=color_texto)
        
        # Dibujar el handle (la bolita grande deslizable)
        x_handle = inicio_x + (self.filtro_index * ancho_seccion) + (ancho_seccion / 2)
        
        # Sombra del handle
        self.slider_canvas.create_oval(x_handle - 11, 21 - 11, x_handle + 11, 21 + 11,
                                      fill="#d0d0d0", outline="")
        
        # Handle principal
        self.slider_canvas.create_oval(x_handle - 10, 20 - 10, x_handle + 10, 20 + 10,
                                      fill=colores[self.filtro_index], 
                                      outline="white", width=3, tags="handle")
        
        # Punto interior del handle
        self.slider_canvas.create_oval(x_handle - 3, 20 - 3, x_handle + 3, 20 + 3,
                                      fill="white", outline="")
        
        # Cambiar cursor cuando esté sobre el handle
        self.slider_canvas.config(cursor="hand2")

    def slider_click(self, event):
        """Maneja el clic en el slider - CORREGIDO PARA 5 OPCIONES"""
        x = event.x
        inicio_x = 20
        ancho_seccion = 480 / 5
        
        # Calcular en qué sección se hizo clic
        if inicio_x <= x <= inicio_x + 480:
            nuevo_index = int((x - inicio_x) / ancho_seccion)
            nuevo_index = max(0, min(4, nuevo_index))
            
            if nuevo_index != self.filtro_index:
                self.filtro_index = nuevo_index
                self.dibujar_slider()
                self.aplicar_filtro()
            
            self.slider_arrastrando = True

    def slider_arrastrar(self, event):
        """Maneja el arrastre del slider - CORREGIDO PARA 5 OPCIONES"""
        if not self.slider_arrastrando:
            return
            
        x = event.x
        inicio_x = 20
        ancho_seccion = 480 / 5
        
        # Calcular nueva posición
        if inicio_x <= x <= inicio_x + 480:
            nuevo_index = int((x - inicio_x) / ancho_seccion)
            nuevo_index = max(0, min(4, nuevo_index))
            
            if nuevo_index != self.filtro_index:
                self.filtro_index = nuevo_index
                self.dibujar_slider()
                self.aplicar_filtro()

    def slider_soltar(self, event):
        """Maneja cuando se suelta el mouse"""
        self.slider_arrastrando = False

    def aplicar_filtro(self):
        """Aplica el filtro seleccionado"""
        filtro = self.filtros_opciones[self.filtro_index]
        self.filtro_var.set(filtro)
        
        # Actualizar badge
        self.root.after(0, self._actualizar_badge_filtro, filtro)
        
        # Aplicar filtro a las alertas activas
        self.root.after(0, self.aplicar_filtro_alertas)
        
        # Notificar al controlador
        if hasattr(self, 'filtro_callback'):
            self.root.after(0, self.filtro_callback, filtro)

    def _actualizar_badge_filtro(self, filtro):
        """Actualiza el badge del filtro sin bloquear"""
        self.label_filtro_actual.config(
            text=filtro,
            bg=self.colores_filtro.get(filtro, "#2196F3")
        )

    def set_filtro_callback(self, callback):
        """Establece el callback para el cambio de filtro"""
        self.filtro_callback = callback

    def _crear_panel_alertas(self, parent, tipo):
        """Crea el panel de alertas reutilizable"""
        tk.Label(parent, text="Alertas activas", bg="#f0f0f0", anchor="w", 
                font=("Arial", 10, "bold")).pack(fill="x", pady=(5, 0), padx=5)
        
        # Frame contenedor para las alertas activas y su scrollbar
        alertas_container = tk.Frame(parent)
        alertas_container.pack(fill="x", pady=5, padx=5)
        
        # Crear canvas y scrollbar
        alertas_canvas = tk.Canvas(alertas_container, height=150, bg="#fff5f5", 
                                   highlightthickness=1, highlightbackground="#ffcccc")
        alertas_scrollbar = ttk.Scrollbar(alertas_container, orient="vertical", command=alertas_canvas.yview)
        
        # Frame interior para el contenido
        frame_alertas_activas = tk.Frame(alertas_canvas, bg="#fff5f5")
        
        # Configurar el canvas
        alertas_canvas.configure(yscrollcommand=alertas_scrollbar.set)
        
        # Empaquetar los elementos
        alertas_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        alertas_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        
        # Crear ventana en el canvas con el frame
        canvas_frame = alertas_canvas.create_window((0, 0), window=frame_alertas_activas, 
                                                    anchor="nw")
        
        # Configurar eventos para actualizar el scrolling
        def on_frame_configure(event):
            alertas_canvas.configure(scrollregion=alertas_canvas.bbox("all"))
        
        def on_canvas_configure(event):
            alertas_canvas.itemconfig(canvas_frame, width=event.width)
        
        frame_alertas_activas.bind("<Configure>", on_frame_configure)
        alertas_canvas.bind("<Configure>", on_canvas_configure)

        # Mensaje inicial
        label_vacio = tk.Label(frame_alertas_activas, text="No hay alertas activas", 
                              bg="#fff5f5", fg="#999999", font=("Arial", 10, "italic"))
        label_vacio.pack(pady=20)

        # Guardar referencias
        if tipo == "completa":
            self.frame_alertas_activas_completa = frame_alertas_activas
            self.alertas_canvas_completa = alertas_canvas
            self.label_vacio_completa = label_vacio
        else:
            self.frame_alertas_activas_solo = frame_alertas_activas
            self.alertas_canvas_solo = alertas_canvas
            self.label_vacio_solo = label_vacio

        # Separador
        tk.Frame(parent, height=2, bg="#cccccc").pack(fill="x", pady=10, padx=5)

        tk.Label(parent, text="Historial de Eventos", bg="#f0f0f0", anchor="w",
                font=("Arial", 10, "bold")).pack(fill="x", pady=(5, 0), padx=5)
        
        # Frame contenedor para el Treeview y scrollbars
        tree_frame = tk.Frame(parent)
        tree_frame.pack(fill="both", expand=True, pady=5, padx=5)
        
        # Crear scrollbars
        tree_scroll_y = ttk.Scrollbar(tree_frame)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        tree_scroll_x = ttk.Scrollbar(tree_frame, orient="horizontal")
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configurar Treeview con scrollbars
        columns = ("ID", "Fecha/Hora", "Cámara", "Tipo", "Confianza")
        history_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10,
                                   yscrollcommand=tree_scroll_y.set,
                                   xscrollcommand=tree_scroll_x.set)
        
        # Configurar scrollbars
        tree_scroll_y.config(command=history_tree.yview)
        tree_scroll_x.config(command=history_tree.xview)
        
        # Configurar columnas
        history_tree.heading("ID", text="ID")
        history_tree.heading("Fecha/Hora", text="Fecha/Hora")
        history_tree.heading("Cámara", text="Cámara")
        history_tree.heading("Tipo", text="Tipo")
        history_tree.heading("Confianza", text="Confianza %")
        
        history_tree.column("ID", width=50)
        history_tree.column("Fecha/Hora", width=150)
        history_tree.column("Cámara", width=100)
        history_tree.column("Tipo", width=200)
        history_tree.column("Confianza", width=100)
        
        history_tree.pack(side=tk.LEFT, fill="both", expand=True)

        # Guardar referencias
        if tipo == "completa":
            self.history_tree_completa = history_tree
        else:
            self.history_tree_solo = history_tree

    def agregar_alerta_visual(self, alerta):
        """Agrega una alerta visual en AMBAS pestañas"""
        # Ocultar mensajes de "no hay alertas"
        if hasattr(self, 'label_vacio_completa'):
            self.label_vacio_completa.pack_forget()
        if hasattr(self, 'label_vacio_solo'):
            self.label_vacio_solo.pack_forget()
        
        # Crear alerta en vista completa
        widget_completa = self._crear_widget_alerta(self.frame_alertas_activas_completa, alerta)
        
        # Crear alerta en vista solo alertas
        widget_solo = self._crear_widget_alerta(self.frame_alertas_activas_solo, alerta)
        
        # Guardar referencia de los widgets con el ID de alerta para poder filtrarlos
        if not hasattr(self, 'alertas_widgets'):
            self.alertas_widgets = {}
        
        self.alertas_widgets[alerta["id_alerta"]] = {
            'completa': widget_completa,
            'solo': widget_solo,
            'tipo': alerta['tipo']
        }
        
        # Agregar al historial en AMBOS árboles
        valores = (
            alerta["id_alerta"],
            alerta["fecha_hora"],
            f"Cámara {int(alerta['id_camara'])+1}",
            alerta["tipo"],
            f"{alerta['confianza']:.1f}%"
        )
        item_completa = self.history_tree_completa.insert("", 0, values=valores)
        item_solo = self.history_tree_solo.insert("", 0, values=valores)
        
        # Guardar los items del treeview para poder filtrarlos
        if not hasattr(self, 'historial_items'):
            self.historial_items = {}
        
        self.historial_items[alerta["id_alerta"]] = {
            'item_completa': item_completa,
            'item_solo': item_solo,
            'tipo': alerta['tipo']
        }
        
        # Aplicar filtro actual a todo (alertas activas + historial)
        self.aplicar_filtro_alertas()

    def aplicar_filtro_alertas(self):
        """Aplica el filtro actual a las alertas activas Y al historial"""
        if not hasattr(self, 'alertas_widgets'):
            self.alertas_widgets = {}
        
        filtro_actual = self.filtro_var.get()
        hay_alertas_visibles = False
        
        # Filtrar alertas activas
        for id_alerta, widgets in self.alertas_widgets.items():
            tipo_alerta = widgets['tipo']
            widget_completa = widgets['completa']
            widget_solo = widgets['solo']
            
            # Mostrar u ocultar según el filtro
            if filtro_actual == "Todas" or tipo_alerta == filtro_actual:
                widget_completa.pack(fill="x", padx=5, pady=3)
                widget_solo.pack(fill="x", padx=5, pady=3)
                hay_alertas_visibles = True
            else:
                widget_completa.pack_forget()
                widget_solo.pack_forget()
        
        # Mostrar mensaje si no hay alertas visibles
        if not hay_alertas_visibles:
            if hasattr(self, 'label_vacio_completa'):
                self.label_vacio_completa.pack(pady=20)
            if hasattr(self, 'label_vacio_solo'):
                self.label_vacio_solo.pack(pady=20)
        
        # Filtrar historial usando los items guardados
        if hasattr(self, 'historial_items'):
            for id_alerta, items_info in self.historial_items.items():
                tipo_evento = items_info['tipo']
                item_completa = items_info['item_completa']
                item_solo = items_info['item_solo']
                
                if filtro_actual == "Todas" or tipo_evento == filtro_actual:
                    # Mostrar item
                    try:
                        self.history_tree_completa.reattach(item_completa, '', 0)
                    except:
                        pass
                    try:
                        self.history_tree_solo.reattach(item_solo, '', 0)
                    except:
                        pass
                else:
                    # Ocultar item
                    try:
                        self.history_tree_completa.detach(item_completa)
                    except:
                        pass
                    try:
                        self.history_tree_solo.detach(item_solo)
                    except:
                        pass

    def _crear_widget_alerta(self, parent, alerta):
        """Crea un widget de alerta individual"""
        # Frame de la alerta
        frame_alerta = tk.Frame(parent, bg="white", relief="solid", borderwidth=1)
        frame_alerta.pack(fill="x", padx=5, pady=3)
        
        # Color según tipo
        color_borde = self.colores_filtro.get(alerta["tipo"], "#FF0000")
        frame_alerta.config(highlightbackground=color_borde, highlightthickness=2)
        
        # Contenido
        frame_contenido = tk.Frame(frame_alerta, bg="white")
        frame_contenido.pack(fill="both", expand=True, padx=8, pady=6)
        
        # Icono y título
        frame_header = tk.Frame(frame_contenido, bg="white")
        frame_header.pack(fill="x")
        
        icono = "🚨" if "robo" in alerta["tipo"].lower() else "⚠️"
        tk.Label(frame_header, text=icono, bg="white", font=("Arial", 14)).pack(side=tk.LEFT)
        
        tk.Label(frame_header, text=alerta["tipo"], bg="white", 
                font=("Arial", 10, "bold"), fg=color_borde).pack(side=tk.LEFT, padx=5)
        
        # Info
        tk.Label(frame_contenido, 
                text=f"Cámara {int(alerta['id_camara'])+1} • {alerta['fecha_hora']} • Confianza: {alerta['confianza']:.1f}%",
                bg="white", font=("Arial", 8), fg="#666666").pack(anchor="w")
        
        # Botón revisar
        btn_revisar = tk.Button(frame_contenido, text="Revisar?", bg=color_borde, fg="white",
                               font=("Arial", 8, "bold"), relief="flat", cursor="hand2",
                               command=lambda: self.marcar_revisada_callback(alerta["id_alerta"], frame_alerta))
        btn_revisar.pack(anchor="e", pady=(5, 0))
        
        return frame_alerta

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
        """Crea un feed de video en AMBAS pestañas"""
        num_camaras = len(self.video_captures)
        
        row = num_camaras // 2
        col = num_camaras % 2
        
        frame_completa = self._crear_frame_camara(self.frame_grid_camaras_completa, id_camara, nombre, row, col)
        frame_solo = self._crear_frame_camara(self.frame_grid_camaras_solo, id_camara, nombre, row, col)
        
        self.video_captures[id_camara] = {
            'panel_completa': frame_completa['panel'],
            'panel_solo': frame_solo['panel'],
            'frame_completa': frame_completa['frame'],
            'frame_solo': frame_solo['frame'],
            'btn_eliminar_completa': frame_completa['btn_eliminar'],
            'btn_eliminar_solo': frame_solo['btn_eliminar']
        }

    def _crear_frame_camara(self, parent, id_camara, nombre, row, col):
        """Crea un frame de cámara individual"""
        frame = tk.Frame(parent, borderwidth=2, relief="groove", bg="white", 
                        width=480, height=400)
        frame.grid(row=row, column=col, padx=10, pady=10)
        frame.grid_propagate(False)
        
        header_frame = tk.Frame(frame, bg="white", height=35)
        header_frame.pack(fill="x", padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        label_nombre = tk.Label(header_frame, text=nombre, bg="white", font=("Arial", 10, "bold"))
        label_nombre.pack(side=tk.LEFT)
        
        btn_eliminar = tk.Button(header_frame, text="✕", bg="red", fg="white", 
                                font=("Arial", 8, "bold"), width=2, height=1,
                                command=lambda: self.eliminar_camara_callback(id_camara))
        btn_eliminar.pack(side=tk.RIGHT)
        
        video_panel = tk.Label(frame, bg="gray15", width=470, height=350)
        video_panel.pack(padx=5, pady=(0, 5))
        
        return {
            'panel': video_panel,
            'frame': frame,
            'btn_eliminar': btn_eliminar
        }

    def eliminar_feed_video(self, id_camara):
        """Elimina un feed de video de AMBAS pestañas"""
        if id_camara in self.video_captures:
            self.video_captures[id_camara]['frame_completa'].destroy()
            self.video_captures[id_camara]['frame_solo'].destroy()
            del self.video_captures[id_camara]
            self.reorganizar_grid()

    def reorganizar_grid(self):
        """Reorganiza todas las cámaras en el grid 2x2"""
        cameras = list(self.video_captures.items())
        for idx, (cam_id, cam_data) in enumerate(cameras):
            row = idx // 2
            col = idx % 2
            cam_data['frame_completa'].grid(row=row, column=col, padx=10, pady=10)
            cam_data['frame_solo'].grid(row=row, column=col, padx=10, pady=10)

    def set_eliminar_callback(self, callback):
        """Establece el callback para eliminar cámaras"""
        self.eliminar_camara_callback = callback

    def set_marcar_revisada_callback(self, callback):
        """Establece el callback para marcar alertas como revisadas"""
        self.marcar_revisada_callback = callback


# CONTROLADOR
class SistemaSeguridadControlador:
    def __init__(self, root):
        self.root = root
        self.modelo = SistemaSeguridad()
        self.vista = SistemaSeguridadVista(root)
        self.monitoreo_activo = False
        self.hilos_camara = {}
        self.capturas_video = {}
        self.contador_camaras = 0
        self.filtro_actual = "Todas"
        
        # Cooldown para alertas (evitar spam)
        self.ultima_alerta_por_tipo = {}
        self.cooldown_segundos = 5

        # Conexión vista-controlador
        self.vista.btn_iniciar.config(command=self.iniciar_monitoreo)
        self.vista.btn_detener.config(command=self.detener_monitoreo)
        self.vista.btn_agregar_camara_completa.config(command=self.agregar_camara)
        self.vista.btn_agregar_camara_solo.config(command=self.agregar_camara)
        self.vista.set_eliminar_callback(self.eliminar_camara)
        self.vista.set_filtro_callback(self.cambiar_filtro_alertas)
        self.vista.set_marcar_revisada_callback(self.marcar_alerta_revisada)

        # Agregar cámaras automáticamente
        self.agregar_camara_automatica()

    def agregar_camara_automatica(self):
        """Intenta agregar cámaras disponibles hasta un máximo de 4"""
        for i in range(4):
            if len(self.modelo.camaras_activas) >= 4:
                break
            self.agregar_camara_real(str(i), f"Cámara {i+1}", f"Ubicación {i+1}")

    def agregar_camara_real(self, id_camara, nombre, ubicacion):
        """Agrega una cámara real al sistema"""
        if len(self.modelo.camaras_activas) >= 4:
            self.vista.mostrar_mensaje("Límite alcanzado", "Solo se permiten 4 cámaras máximo")
            return
            
        if any(cam["id"] == id_camara for cam in self.modelo.camaras_activas):
            return
            
        try:
            cap = cv2.VideoCapture(int(id_camara))
            if not cap.isOpened():
                return
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.capturas_video[id_camara] = cap
            self.modelo.agregar_camara(id_camara, nombre, ubicacion)
            self.vista.crear_feed_video(id_camara, f"{nombre} ({ubicacion})")
            
            if self.monitoreo_activo:
                hilo = threading.Thread(target=self.procesar_video, args=(id_camara,))
                hilo.daemon = True
                hilo.start()
                self.hilos_camara[id_camara] = hilo
                
        except Exception as e:
            pass

    def eliminar_camara(self, id_camara):
        """Elimina una cámara del sistema"""
        if id_camara in self.capturas_video:
            if self.capturas_video[id_camara].isOpened():
                self.capturas_video[id_camara].release()
            del self.capturas_video[id_camara]
        
        self.modelo.eliminar_camara(id_camara)
        self.vista.eliminar_feed_video(id_camara)
        
        if id_camara in self.hilos_camara:
            del self.hilos_camara[id_camara]

    def iniciar_monitoreo(self):
        """Inicia el monitoreo"""
        if self.monitoreo_activo:
            return
        self.monitoreo_activo = True
        self.modelo.estado_sistema = "Monitoreo"
        self.vista.actualizar_estado("Monitoreo")

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
        """Detiene el monitoreo"""
        if not self.monitoreo_activo:
            return
        
        self.monitoreo_activo = False
        self.modelo.estado_sistema = "Detenido"
        self.vista.actualizar_estado("Detenido")
        time.sleep(0.2)
        self.vista.mostrar_mensaje("Monitoreo", "El monitoreo se ha detenido")

    def puede_generar_alerta(self, id_camara, tipo_alerta):
        """Verifica si puede generar una alerta (cooldown)"""
        clave = f"{id_camara}_{tipo_alerta}"
        ahora = time.time()
        
        if clave in self.ultima_alerta_por_tipo:
            tiempo_transcurrido = ahora - self.ultima_alerta_por_tipo[clave]
            if tiempo_transcurrido < self.cooldown_segundos:
                return False
        
        self.ultima_alerta_por_tipo[clave] = ahora
        return True

    def procesar_detecciones(self, detecciones, id_camara):
        """Procesa las detecciones y genera alertas"""
        if not detecciones:
            return
        
        for deteccion in detecciones:
            tipo = deteccion.get('tipo', 'Desconocido')
            confianza = deteccion.get('confianza', 0) * 100
            
            # Ignorar detecciones de "persona" normal
            if tipo.lower() in ['persona', 'person', 'people']:
                continue
            
            # Mapear tipos de detección a tipos de alerta
            tipo_alerta = self.mapear_tipo_alerta(tipo)
            
            # Verificar si debe mostrar según filtro actual
            if self.filtro_actual != "Todas" and tipo_alerta != self.filtro_actual:
                continue
            
            # Verificar cooldown
            if not self.puede_generar_alerta(id_camara, tipo_alerta):
                continue
            
            # Crear alerta en el modelo
            alerta = self.modelo.agregar_alerta(id_camara, tipo_alerta, confianza)
            
            # Mostrar en la vista
            self.vista.root.after(0, self.vista.agregar_alerta_visual, alerta)
            
            print(f"🚨 ALERTA: {tipo_alerta} detectado en Cámara {int(id_camara)+1} (Confianza: {confianza:.1f}%)")

    def mapear_tipo_alerta(self, tipo_deteccion):
        """Mapea el tipo de detección a tipo de alerta"""
        tipo_lower = tipo_deteccion.lower()
        
        if 'sosteniendo' in tipo_lower or 'holding' in tipo_lower:
            return "Sosteniendo mercancía"
        elif 'sospechoso' in tipo_lower or 'suspicious' in tipo_lower:
            return "Comportamiento sospechoso"
        elif 'escondiendo' in tipo_lower or 'hiding' in tipo_lower:
            return "Escondiendo mercancía"
        elif 'robo' in tipo_lower or 'theft' in tipo_lower or 'stealing' in tipo_lower:
            return "Posible robo"
        else:
            return "Comportamiento sospechoso"

    def procesar_video(self, id_camara):
        """Procesa el video de una cámara con detección"""
        try:
            from proto1 import detectar_personas
            usar_deteccion = True
        except:
            usar_deteccion = False
            
        while self.monitoreo_activo and id_camara in self.capturas_video:
            cap = self.capturas_video[id_camara]
            
            if not cap.isOpened():
                break
                
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            detecciones = None
            
            # Aplicar detección si está disponible
            if usar_deteccion:
                try:
                    frame, detecciones = detectar_personas(frame)
                    
                    # Procesar detecciones y generar alertas
                    if detecciones:
                        self.procesar_detecciones(detecciones, id_camara)
                        
                except:
                    usar_deteccion = False

            if id_camara in self.vista.video_captures:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    img_resized = img.resize((470, 350), Image.Resampling.LANCZOS)
                    img_tk = ImageTk.PhotoImage(image=img_resized)
                    
                    self.vista.root.after_idle(self.actualizar_label_video, id_camara, img_tk)
                except:
                    pass

            time.sleep(0.03)

    def actualizar_label_video(self, id_camara, img_tk):
        """Actualiza el video en AMBAS pestañas"""
        if id_camara in self.vista.video_captures:
            label_completa = self.vista.video_captures[id_camara]['panel_completa']
            label_completa.imgtk = img_tk
            label_completa.configure(image=img_tk)
            
            label_solo = self.vista.video_captures[id_camara]['panel_solo']
            label_solo.imgtk = img_tk
            label_solo.configure(image=img_tk)

    def agregar_camara(self):
        """Permite agregar una nueva cámara manualmente"""
        if len(self.modelo.camaras_activas) >= 4:
            self.vista.mostrar_mensaje("Límite alcanzado", "Solo se permiten 4 cámaras máximo")
            return
            
        for i in range(10):
            id_camara = str(i)
            if not any(cam["id"] == id_camara for cam in self.modelo.camaras_activas):
                self.agregar_camara_real(id_camara, f"Cámara {i+1}", f"Ubicación {i+1}")
                return
                
        self.vista.mostrar_mensaje("Error", "No se encontraron más cámaras disponibles")

    def cambiar_filtro_alertas(self, filtro):
        """Maneja el cambio de filtro de alertas"""
        self.filtro_actual = filtro

    def marcar_alerta_revisada(self, id_alerta, frame_widget):
        """Marca una alerta como revisada"""
        if self.modelo.marcar_alerta_revisada(id_alerta):
            # Animar y eliminar el widget
            frame_widget.config(bg="#e8f5e9")
            
            # Eliminar de la lista de widgets activos
            if hasattr(self.vista, 'alertas_widgets') and id_alerta in self.vista.alertas_widgets:
                del self.vista.alertas_widgets[id_alerta]
            
            # Destruir después de la animación
            self.root.after(300, lambda: self._destruir_alerta(frame_widget))
    
    def _destruir_alerta(self, widget):
        """Destruye el widget de alerta y verifica si hay que mostrar mensaje vacío"""
        try:
            widget.destroy()
            
            # Verificar si hay alertas visibles después de eliminar
            if hasattr(self.vista, 'alertas_widgets'):
                filtro_actual = self.vista.filtro_var.get()
                hay_visibles = False
                
                for id_alerta, widgets_info in self.vista.alertas_widgets.items():
                    if filtro_actual == "Todas" or widgets_info['tipo'] == filtro_actual:
                        hay_visibles = True
                        break
                
                # Si no hay alertas visibles, mostrar mensaje
                if not hay_visibles:
                    if hasattr(self.vista, 'label_vacio_completa'):
                        self.vista.label_vacio_completa.pack(pady=20)
                    if hasattr(self.vista, 'label_vacio_solo'):
                        self.vista.label_vacio_solo.pack(pady=20)
        except:
            pass

    def __del__(self):
        """Limpieza al cerrar"""
        try:
            self.monitoreo_activo = False
            time.sleep(0.3)
            for cap in self.capturas_video.values():
                if cap.isOpened():
                    cap.release()
        except:
            pass


# MAIN
if __name__ == "__main__":
    root = tk.Tk()
    app = SistemaSeguridadControlador(root)
    root.mainloop()
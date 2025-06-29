import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Ruta del video
video_path = "VIDEOS DE STOCK/video1.mp4"

# Cargar modelo YOLOv8 (más preciso que v5)
model = YOLO("yolov8n.pt")  # Cambiar a yolov8s.pt o yolov8m.pt para mayor precisión

# DeepSort con parámetros más estrictos
tracker = DeepSort(
    max_age=30,         # Reducido para eliminar tracks fantasma más rápido
    n_init=5,           # Más frames necesarios para confirmar track
    nn_budget=50,       # Reducido para mejor rendimiento
    max_cosine_distance=0.2  # Más estricto en la asociación
)

# Filtros de detección más estrictos
conf_threshold = 0.65      # Aumentado significativamente
nms_threshold = 0.3        # Non-Maximum Suppression más agresivo
min_area = 15000           # Área mínima más grande
max_area = 400000          # Área máxima para evitar detecciones erróneas
min_aspect_ratio = 1.2     # Personas son más altas que anchas
max_aspect_ratio = 4.0     # Límite superior realista
min_area_percent = 0.025   # Porcentaje mínimo de área del frame

# Captura del video
cap = cv2.VideoCapture(video_path)
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Región de interés (ROI) - opcional, ajustar según necesidad
roi_y_start = int(video_height * 0.1)  # Ignorar 10% superior
roi_y_end = int(video_height * 0.95)   # Ignorar 5% inferior

# Inicializar grabación con mejor codec
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "video_resultado_optimizado.mp4"
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))

# Ventana de visualización
cv2.namedWindow("Detector Optimizado", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detector Optimizado", 1280, 720)

# Funciones auxiliares mejoradas
def calculate_iou(boxA, boxB):
    """Calcula Intersection over Union más eficientemente"""
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    areaA = (xa2 - xa1) * (ya2 - ya1)
    areaB = (xb2 - xb1) * (yb2 - yb1)
    union = areaA + areaB - inter_area
    
    return inter_area / union if union > 0 else 0

def is_valid_person_detection(box, frame_width, frame_height):
    """Valida si una detección es una persona válida"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect_ratio = height / width if width > 0 else 0
    area_percent = area / (frame_width * frame_height)
    
    # Verificar límites básicos
    if area < min_area or area > max_area:
        return False
    
    if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
        return False
    
    if area_percent < min_area_percent:
        return False
    
    # Verificar que la caja no esté en los bordes extremos
    border_margin = 20
    if (x1 < border_margin or y1 < border_margin or 
        x2 > frame_width - border_margin or y2 > frame_height - border_margin):
        return False
    
    # Verificar que esté dentro del ROI
    if y1 < roi_y_start or y2 > roi_y_end:
        return False
    
    return True

def refine_bounding_box(box, margin_factor=0.05):
    """Ajusta la bounding box para ceñirla mejor"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Reducir ligeramente la caja para ceñir mejor
    margin_x = width * margin_factor
    margin_y = height * margin_factor
    
    return [x1 + margin_x, y1 + margin_y, x2 - margin_x, y2 - margin_y]

def non_max_suppression_custom(detections, iou_threshold=0.3):
    """NMS personalizado para eliminar detecciones superpuestas"""
    if not detections:
        return []
    
    # Ordenar por confianza
    detections.sort(key=lambda x: x[1], reverse=True)
    
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        detections = [det for det in detections 
                     if calculate_iou(current[0], det[0]) < iou_threshold]
    
    return keep

# Contador de frames para estabilidad
frame_count = 0
detection_history = []  # Historial para suavizado temporal

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Procesar cada N frames para mejor rendimiento
    if frame_count % 1 == 0:  # Cambiar a 2 o 3 para saltar frames
        # Ejecutar detección en resolución original para mejor precisión
        results = model(frame, conf=conf_threshold, iou=nms_threshold, verbose=False)
        
        raw_detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, class_id in zip(boxes, scores, class_ids):
                # Solo personas (class_id == 0)
                if class_id == 0 and score > conf_threshold:
                    x1, y1, x2, y2 = box
                    
                    # Validar detección
                    if is_valid_person_detection([x1, y1, x2, y2], video_width, video_height):
                        # Refinar bounding box
                        refined_box = refine_bounding_box([x1, y1, x2, y2])
                        raw_detections.append((refined_box, score, 'person'))
        
        # Aplicar NMS personalizado
        filtered_detections = non_max_suppression_custom(raw_detections, 0.3)
        
        # Filtrado adicional por coherencia temporal
        if len(detection_history) > 5:
            detection_history.pop(0)
        detection_history.append(filtered_detections)
        
        # Usar detecciones actuales para tracking
        current_detections = filtered_detections
    else:
        # Usar detecciones del frame anterior
        current_detections = detection_history[-1] if detection_history else []
    
    # Seguimiento con DeepSort
    tracks = tracker.update_tracks(current_detections, frame=frame)
    
    # Dibujar resultados
    active_tracks = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        active_tracks += 1
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        
        # Asegurar que las coordenadas estén dentro del frame
        x1 = max(0, min(x1, video_width))
        y1 = max(0, min(y1, video_height))
        x2 = max(0, min(x2, video_width))
        y2 = max(0, min(y2, video_height))
        
        # Dibujar bounding box más elegante
        color = (0, 255, 0)  # Verde
        thickness = 2
        
        # Rectángulo principal
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Esquinas para mejor visualización
        corner_length = 20
        corner_thickness = 3
        
        # Esquina superior izquierda
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, corner_thickness)
        
        # Esquina superior derecha
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, corner_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, corner_thickness)
        
        # Esquina inferior izquierda
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, corner_thickness)
        
        # Esquina inferior derecha
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, corner_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, corner_thickness)
        
        # Etiqueta mejorada
        label_text = f"ID:{track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Calcular tamaño del texto
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        
        # Posición de la etiqueta (arriba de la caja)
        label_x = x1
        label_y = y1 - 10 if y1 > 30 else y2 + 25
        
        # Fondo de la etiqueta
        cv2.rectangle(frame, 
                     (label_x, label_y - text_height - 5),
                     (label_x + text_width + 10, label_y + 5),
                     color, -1)
        
        # Texto de la etiqueta
        cv2.putText(frame, label_text, (label_x + 5, label_y), 
                   font, font_scale, (0, 0, 0), font_thickness)
    
    # Información en pantalla
    info_text = f"Personas detectadas: {active_tracks} | Frame: {frame_count}"
    cv2.putText(frame, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Mostrar y grabar
    cv2.imshow("Detector Optimizado", frame)
    video_writer.write(frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpieza
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Video procesado guardado en: {output_path}")
print(f"Total de frames procesados: {frame_count}")
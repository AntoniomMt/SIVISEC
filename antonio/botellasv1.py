"""
Detector Robusto de Persona Sosteniendo Botella
Utiliza YOLOv8 para detección de objetos y MediaPipe para orientación de manos
Versión optimizada con tracking de centroides y validación de agarre
"""

import cv2
from ultralytics import YOLO
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from collections import deque
import time

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HandOrientationDetector:
    """Detector de orientación de manos usando MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=4,  # Reducir máximo de manos
            min_detection_confidence=0.5,
            min_tracking_confidence=0.4,
            model_complexity=0  # Usar modelo más simple (0 = lite, 1 = full)
        )
    
    def calcular_orientacion_mano(self, landmarks) -> str:
        """
        Calcula la orientación de la mano (palma, dorso o rotación)
        CORREGIDO: Lógica invertida para detección correcta
        """
        # Puntos clave
        wrist = landmarks[0]
        index_tip = landmarks[8]
        index_mcp = landmarks[5]
        index_pip = landmarks[6]
        middle_tip = landmarks[12]
        middle_mcp = landmarks[9]
        middle_pip = landmarks[10]
        pinky_tip = landmarks[20]
        pinky_mcp = landmarks[17]
        pinky_pip = landmarks[18]
        
        # Calcular diferencias en Z (profundidad)
        # Si z es POSITIVO = más lejos de la cámara
        # Si z es NEGATIVO = más cerca de la cámara
        index_diff = index_tip.z - index_mcp.z
        middle_diff = middle_tip.z - middle_mcp.z
        pinky_diff = pinky_tip.z - pinky_mcp.z
        
        index_pip_diff = index_pip.z - index_mcp.z
        middle_pip_diff = middle_pip.z - middle_mcp.z
        pinky_pip_diff = pinky_pip.z - pinky_mcp.z
        
        avg_diff = (index_diff + middle_diff + pinky_diff) / 3
        avg_pip_diff = (index_pip_diff + middle_pip_diff + pinky_pip_diff) / 3
        
        # Vectores para detectar rotación
        palm_x = middle_mcp.x - wrist.x
        palm_y = middle_mcp.y - wrist.y
        horizontal_x = index_mcp.x - pinky_mcp.x
        horizontal_y = index_mcp.y - pinky_mcp.y
        
        h_length = np.sqrt(horizontal_x**2 + horizontal_y**2)
        v_length = np.sqrt(palm_x**2 + palm_y**2)
        rotation_ratio = h_length / (v_length + 0.001)
        
        # CORREGIDO: Clasificar orientación con lógica correcta
        if rotation_ratio < 0.4:
            return 'rotacion'
        # Si las puntas están MÁS CERCA (z menor) que nudillos = PALMA visible
        elif avg_diff < -0.01 and avg_pip_diff < -0.005:
            return 'palma'
        # Si las puntas están MÁS LEJOS (z mayor) que nudillos = DORSO visible
        else:
            return 'dorso'
    
    def obtener_centroide_mano(self, landmarks, img_width: int, img_height: int) -> Tuple[int, int]:
        """
        Calcula el centroide de la mano
        """
        x_coords = [lm.x * img_width for lm in landmarks]
        y_coords = [lm.y * img_height for lm in landmarks]
        
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        return centroid_x, centroid_y
    
    def detectar_manos(self, frame: np.ndarray) -> List[Dict]:
        """
        Detecta manos y retorna información de orientación y posición
        OPTIMIZADO: Procesa frame reducido para mejor performance
        """
        # Reducir resolución para MediaPipe (gran mejora de FPS)
        frame_height, frame_width = frame.shape[:2]
        scale = 0.5  # Procesar a mitad de resolución
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        manos_detectadas = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                orientacion = self.calcular_orientacion_mano(hand_landmarks.landmark)
                
                # Escalar coordenadas de vuelta al tamaño original
                x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
                y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
                
                centroid_x = int(np.mean(x_coords))
                centroid_y = int(np.mean(y_coords))
                
                manos_detectadas.append({
                    'orientacion': orientacion,
                    'centroid': (centroid_x, centroid_y),
                    'landmarks': hand_landmarks.landmark
                })
        
        return manos_detectadas
    
    def close(self):
        self.hands.close()


class BottleTracker:
    """Clase para trackear botellas y sus relaciones con manos"""
    
    def __init__(self, collision_history_size: int = 10, distance_threshold: int = 80):
        self.collision_history = {}  # {bottle_id: deque de colisiones}
        self.collision_history_size = collision_history_size
        self.distance_threshold = distance_threshold  # Umbral de distancia en píxeles
        self.next_bottle_id = 0
        self.bottle_positions = {}  # {bottle_id: (x, y)}
    
    def calcular_centroide_botella(self, bbox: List[float]) -> Tuple[int, int]:
        """Calcula el centroide de una botella"""
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 2)
        return x_center, y_center
    
    def encontrar_botella_cercana(self, centroid: Tuple[int, int]) -> Optional[int]:
        """Encuentra la botella más cercana a una posición"""
        min_dist = float('inf')
        closest_id = None
        
        for bottle_id, pos in self.bottle_positions.items():
            dist = np.sqrt((pos[0] - centroid[0])**2 + (pos[1] - centroid[1])**2)
            if dist < min_dist and dist < 150:  # Umbral máximo para asociación
                min_dist = dist
                closest_id = bottle_id
        
        return closest_id
    
    def asignar_id_botella(self, centroid: Tuple[int, int]) -> int:
        """Asigna o recupera el ID de una botella basado en su posición"""
        bottle_id = self.encontrar_botella_cercana(centroid)
        
        if bottle_id is None:
            bottle_id = self.next_bottle_id
            self.next_bottle_id += 1
            self.collision_history[bottle_id] = deque(maxlen=self.collision_history_size)
        
        self.bottle_positions[bottle_id] = centroid
        return bottle_id
    
    def registrar_colision(self, bottle_id: int, tiene_colision: bool):
        """Registra si hay colisión en este frame"""
        if bottle_id not in self.collision_history:
            self.collision_history[bottle_id] = deque(maxlen=self.collision_history_size)
        self.collision_history[bottle_id].append(tiene_colision)
    
    def ha_tenido_colision(self, bottle_id: int) -> bool:
        """Verifica si la botella ha tenido al menos una colisión en su historia"""
        if bottle_id not in self.collision_history:
            return False
        return any(self.collision_history[bottle_id])
    
    def limpiar_ids_antiguos(self):
        """Limpia IDs de botellas que ya no están presentes"""
        ids_actuales = set(self.bottle_positions.keys())
        ids_historicos = set(self.collision_history.keys())
        
        # Eliminar IDs que llevan mucho tiempo sin actualizarse
        for bottle_id in list(ids_historicos - ids_actuales):
            if len(self.collision_history[bottle_id]) == 0:
                del self.collision_history[bottle_id]


class BottlePersonDetector:
    """Clase mejorada para detectar si una persona está sosteniendo una botella"""
    
    def __init__(self, model_name: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Inicializa el detector
        
        Args:
            model_name: Nombre del modelo YOLO a utilizar
            confidence_threshold: Umbral de confianza para las detecciones
        """
        try:
            self.model = YOLO(model_name)
            self.confidence_threshold = confidence_threshold
            
            # IDs de clases COCO
            self.PERSON_CLASS_ID = 0
            self.BOTTLE_CLASS_ID = 39
            
            # Detector de orientación de manos
            self.hand_detector = HandOrientationDetector()
            
            # Tracker de botellas
            self.bottle_tracker = BottleTracker(collision_history_size=10, distance_threshold=80)
            
            # FPS tracking
            self.fps_history = deque(maxlen=15)
            self.last_time = time.time()
            
            # Frame skipping para optimización
            self.frame_count = 0
            self.process_every_n_frames = 1  # Procesar cada N frames
            self.last_detections = []
            
            logger.info(f"Modelo {model_name} cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise
    
    def calcular_fps(self) -> float:
        """Calcula los FPS actuales"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_history.append(fps)
        return np.mean(self.fps_history)
    
    def distancia_euclidiana(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calcula la distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], float]:
        """
        Detecta personas y botellas con validación de agarre
        OPTIMIZADO: Frame skipping y procesamiento reducido
        
        Args:
            frame: Imagen en formato numpy array (BGR)
            
        Returns:
            Tupla con la imagen anotada, lista de detecciones y FPS
        """
        # Calcular FPS
        fps = self.calcular_fps()
        
        # Incrementar contador de frames
        self.frame_count += 1
        
        # Skip frames para mejorar FPS (opcional)
        # if self.frame_count % self.process_every_n_frames != 0:
        #     return frame, self.last_detections, fps
        
        # Redimensionar frame para mejorar rendimiento
        frame_height, frame_width = frame.shape[:2]
        scale_factor = 0.75  # Más agresivo para mejor FPS
        if frame_width > 960:
            scale_factor = 960 / frame_width
        
        frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Detectar con YOLO (solo botellas para optimizar)
        results = self.model.track(frame_resized, 
                                   classes=[self.BOTTLE_CLASS_ID],  # Solo botellas
                                   conf=self.confidence_threshold, 
                                   verbose=False,
                                   persist=True)
        
        bottles = []
        
        # Extraer detecciones y escalar de vuelta
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # Escalar bbox al tamaño original
                bbox = [coord / scale_factor for coord in bbox]
                bottles.append({'bbox': bbox, 'confidence': confidence})
        
        # Detectar manos y orientaciones
        manos = self.hand_detector.detectar_manos(frame)
        
        # Analizar botellas
        annotated_frame = frame.copy()
        detections = []
        
        for bottle in bottles:
            bottle_centroid = self.bottle_tracker.calcular_centroide_botella(bottle['bbox'])
            bottle_id = self.bottle_tracker.asignar_id_botella(bottle_centroid)
            
            # Verificar si alguna mano válida está cerca
            tiene_colision_actual = False
            mano_sosteniendo = None
            
            for mano in manos:
                # Filtrar manos con palma visible (no pueden sostener correctamente)
                if mano['orientacion'] == 'palma':
                    continue
                
                # Calcular distancia entre centroides
                distancia = self.distancia_euclidiana(mano['centroid'], bottle_centroid)
                
                # Verificar colisión
                if distancia < self.bottle_tracker.distance_threshold:
                    tiene_colision_actual = True
                    mano_sosteniendo = mano
                    break
            
            # Registrar colisión
            self.bottle_tracker.registrar_colision(bottle_id, tiene_colision_actual)
            
            # Determinar si está siendo sostenida
            esta_siendo_sostenida = (
                tiene_colision_actual and 
                self.bottle_tracker.ha_tenido_colision(bottle_id)
            )
            
            # Dibujar botella
            bx1, by1, bx2, by2 = map(int, bottle['bbox'])
            color_botella = (0, 255, 0) if esta_siendo_sostenida else (0, 165, 255)
            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color_botella, 2)
            
            # Dibujar centroide de botella
            cv2.circle(annotated_frame, bottle_centroid, 5, color_botella, -1)
            
            # Etiqueta simple
            label = "SOSTENIDA" if esta_siendo_sostenida else "SUELTA"
            cv2.putText(annotated_frame, label, (bx1, by1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_botella, 2)
            
            # Si está siendo sostenida, dibujar línea a la mano
            if esta_siendo_sostenida and mano_sosteniendo:
                cv2.line(annotated_frame, bottle_centroid, 
                        mano_sosteniendo['centroid'], (0, 255, 0), 2)
            
            detections.append({
                'bottle_id': bottle_id,
                'bottle_bbox': bottle['bbox'],
                'bottle_confidence': bottle['confidence'],
                'bottle_centroid': bottle_centroid,
                'being_held': esta_siendo_sostenida,
                'has_collision_history': self.bottle_tracker.ha_tenido_colision(bottle_id)
            })
        
        # Dibujar manos con sus orientaciones (más simple)
        for mano in manos:
            color_mano = {
                'palma': (144, 238, 144),    # Verde claro - NO VÁLIDA
                'dorso': (255, 144, 30),      # Naranja - VÁLIDA
                'rotacion': (0, 255, 255)     # Amarillo - VÁLIDA
            }.get(mano['orientacion'], (255, 255, 255))
            
            # Dibujar centroide de mano más pequeño
            cv2.circle(annotated_frame, mano['centroid'], 5, color_mano, -1)
            cv2.circle(annotated_frame, mano['centroid'], 8, color_mano, 2)
            
            # Etiqueta compacta
            cv2.putText(annotated_frame, mano['orientacion'][0].upper(), 
                       (mano['centroid'][0] + 12, mano['centroid'][1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_mano, 1)
        
        # Limpiar tracker
        self.bottle_tracker.limpiar_ids_antiguos()
        
        # Guardar detecciones para frame skipping
        self.last_detections = detections
        
        return annotated_frame, detections, fps
    
    def process_video(self, source: int = 0, output_file: str = None, show_debug: bool = True):
        """
        Procesa video en tiempo real desde webcam o archivo
        
        Args:
            source: 0 para webcam, o ruta de archivo de video
            output_file: Ruta para guardar el video procesado (opcional)
            show_debug: Mostrar información de debug
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error("No se pudo abrir la fuente de video")
            return
        
        # Configurar captura para mejor rendimiento
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Configurar writer si se desea guardar
        writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        logger.info("=== CONTROLES ===")
        logger.info("'q' - Salir")
        logger.info("=== INFO ===")
        logger.info("Verde: Botella sostenida | Naranja: Botella suelta")
        logger.info("Manos: Verde claro=Palma(no valida) | Naranja=Dorso | Amarillo=Rotacion")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Voltear para efecto espejo
                frame = cv2.flip(frame, 1)
                
                # Procesar frame
                annotated_frame, detections, fps = self.detect(frame)
                
                # Estadísticas simplificadas - solo FPS y conteo
                bottles_held = sum(1 for d in detections if d['being_held'])
                total_bottles = len(detections)
                
                # FPS en esquina superior derecha
                fps_text = f"FPS: {fps:.1f}"
                text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = annotated_frame.shape[1] - text_size[0] - 10
                
                # Fondo para mejor legibilidad
                cv2.rectangle(annotated_frame, 
                            (text_x - 5, 10), 
                            (annotated_frame.shape[1] - 5, 45), 
                            (0, 0, 0), -1)
                cv2.putText(annotated_frame, fps_text, (text_x, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Conteo de botellas en la izquierda
                stats_text = f"Botellas: {total_bottles} | Sostenidas: {bottles_held}"
                cv2.rectangle(annotated_frame, (5, 10), (380, 45), (0, 0, 0), -1)
                cv2.putText(annotated_frame, stats_text, (10, 32),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostrar resultado
                cv2.imshow('Detector Avanzado - Botella Sostenida', annotated_frame)
                
                # Guardar si es necesario
                if writer:
                    writer.write(annotated_frame)
                
                # Controles simplificados
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrumpido por el usuario")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.hand_detector.close()
            logger.info("Procesamiento finalizado")
    
    def process_image(self, image_path: str, output_path: str = None) -> List[Dict]:
        """
        Procesa una imagen estática
        
        Args:
            image_path: Ruta de la imagen a procesar
            output_path: Ruta para guardar la imagen procesada (opcional)
            
        Returns:
            Lista de detecciones
        """
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"No se pudo leer la imagen: {image_path}")
            return []
        
        annotated_frame, detections, fps = self.detect(frame)
        
        # Mostrar FPS en la imagen
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                   (annotated_frame.shape[1] - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Mostrar y/o guardar
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            logger.info(f"Imagen guardada en: {output_path}")
        
        cv2.imshow('Resultado', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self.hand_detector.close()
        
        return detections


def main():
    """Función principal para demostración"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detector Avanzado de Persona Sosteniendo Botella')
    parser.add_argument('--mode', type=str, choices=['webcam', 'video', 'image'], 
                       default='webcam', help='Modo de operación')
    parser.add_argument('--source', type=str, help='Ruta del archivo de video o imagen')
    parser.add_argument('--output', type=str, help='Ruta para guardar el resultado')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Umbral de confianza (0-1)')
    parser.add_argument('--no-debug', action='store_true', 
                       help='Desactivar información de debug')
    
    args = parser.parse_args()
    
    # Crear detector
    detector = BottlePersonDetector(confidence_threshold=args.confidence)
    
    # Ejecutar según el modo
    if args.mode == 'webcam':
        detector.process_video(source=0, output_file=args.output, show_debug=not args.no_debug)
    elif args.mode == 'video':
        if not args.source:
            logger.error("Debes especificar --source para el modo video")
            return
        detector.process_video(source=args.source, output_file=args.output, show_debug=not args.no_debug)
    elif args.mode == 'image':
        if not args.source:
            logger.error("Debes especificar --source para el modo image")
            return
        detections = detector.process_image(args.source, args.output)
        
        # Imprimir resultados
        for i, det in enumerate(detections, 1):
            status = "SÍ" if det['being_held'] else "NO"
            logger.info(f"Botella {i}: {status} está siendo sostenida")


if __name__ == "__main__":
    main()

import cv2
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BottlePersonDetector:
    """Clase para detectar si una persona está sosteniendo una botella"""
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
            
            # Umbral de proximidad (en píxeles) para considerar que una botella está cerca de una persona
            self.proximity_threshold = 100
            
            logger.info(f"Modelo {model_name} cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise
    
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calcula el Intersection over Union (IoU) entre dos cajas delimitadoras
        Args:
            box1, box2: Listas con [x1, y1, x2, y2]
        Returns:
            float: Valor de IoU entre 0 y 1
        """
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def is_bottle_near_person(self, person_box: List[float], bottle_box: List[float]) -> bool:
        """
        Determina si una botella está cerca o dentro del área de una persona
        Args:
            person_box: Caja delimitadora de la persona [x1, y1, x2, y2]
            bottle_box: Caja delimitadora de la botella [x1, y1, x2, y2]
        Returns:
            bool: True si la botella está cerca de la persona
        """
        # Verificar IoU (superposición)
        iou = self.calculate_iou(person_box, bottle_box)
        if iou > 0.1:  # Si hay superposición significativa
            return True
        
        # Verificar proximidad del centro de la botella con la persona
        bottle_center_x = (bottle_box[0] + bottle_box[2]) / 2
        bottle_center_y = (bottle_box[1] + bottle_box[3]) / 2
        
        # Expandir el área de la persona
        person_expanded = [
            person_box[0] - self.proximity_threshold,
            person_box[1] - self.proximity_threshold,
            person_box[2] + self.proximity_threshold,
            person_box[3] + self.proximity_threshold
        ]
        
        # Verificar si el centro de la botella está dentro del área expandida
        if (person_expanded[0] <= bottle_center_x <= person_expanded[2] and
            person_expanded[1] <= bottle_center_y <= person_expanded[3]):
            return True
        
        return False
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detecta personas y botellas en un frame
        Args:
            frame: Imagen en formato numpy array (BGR)
        Returns:
            Tupla con la imagen anotada y lista de detecciones
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        persons = []
        bottles = []
        detections = []
        
        # Extraer detecciones
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                if class_id == self.PERSON_CLASS_ID:
                    persons.append({'bbox': bbox, 'confidence': confidence})
                elif class_id == self.BOTTLE_CLASS_ID:
                    bottles.append({'bbox': bbox, 'confidence': confidence})
        
        # Analizar relaciones persona-botella
        annotated_frame = frame.copy()
        
        for person in persons:
            has_bottle = False
            person_bottles = []
            
            for bottle in bottles:
                if self.is_bottle_near_person(person['bbox'], bottle['bbox']):
                    has_bottle = True
                    person_bottles.append(bottle)
            
            # Dibujar caja de la persona
            x1, y1, x2, y2 = map(int, person['bbox'])
            color = (0, 255, 0) if has_bottle else (0, 165, 255)  # Verde si tiene botella, naranja si no
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta
            label = f"Persona con botella ({person['confidence']:.2f})" if has_bottle else f"Persona ({person['confidence']:.2f})"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Dibujar botellas asociadas
            for bottle in person_bottles:
                bx1, by1, bx2, by2 = map(int, bottle['bbox'])
                cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Botella ({bottle['confidence']:.2f})", 
                           (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            detections.append({
                'person_bbox': person['bbox'],
                'person_confidence': person['confidence'],
                'has_bottle': has_bottle,
                'bottles': person_bottles
            })
        
        return annotated_frame, detections
    
    def process_video(self, source: int = 0, output_file: str = None):
        """
        Procesa video en tiempo real desde webcam o archivo
        Args:
            source: 0 para webcam, o ruta de archivo de video
            output_file: Ruta para guardar el video procesado (opcional)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error("No se pudo abrir la fuente de video")
            return
        
        # Configurar writer si se desea guardar
        writer = None
        if output_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        logger.info("Presiona 'q' para salir")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Procesar frame
                annotated_frame, detections = self.detect(frame)
                
                # Mostrar estadísticas
                people_with_bottles = sum(1 for d in detections if d['has_bottle'])
                total_people = len(detections)
                
                stats_text = f"Personas: {total_people} | Con botella: {people_with_bottles}"
                cv2.putText(annotated_frame, stats_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Mostrar resultado
                cv2.imshow('Detector de Persona con Botella', annotated_frame)
                
                # Guardar si es necesario
                if writer:
                    writer.write(annotated_frame)
                
                # Salir con 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrumpido por el usuario")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
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
        
        annotated_frame, detections = self.detect(frame)
        
        # Mostrar y/o guardar
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            logger.info(f"Imagen guardada en: {output_path}")
        
        cv2.imshow('Resultado', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detections


def main():
    """Función principal para demostración"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detector de Persona Sosteniendo Botella')
    parser.add_argument('--mode', type=str, choices=['webcam', 'video', 'image'], 
                       default='webcam', help='Modo de operación')
    parser.add_argument('--source', type=str, help='Ruta del archivo de video o imagen')
    parser.add_argument('--output', type=str, help='Ruta para guardar el resultado')
    parser.add_argument('--confidence', type=float, default=0.5, 
                       help='Umbral de confianza (0-1)')
    
    args = parser.parse_args()
    
    # Crear detector
    detector = BottlePersonDetector(confidence_threshold=args.confidence)
    
    # Ejecutar según el modo
    if args.mode == 'webcam':
        detector.process_video(source=0, output_file=args.output)
    elif args.mode == 'video':
        if not args.source:
            logger.error("Debes especificar --source para el modo video")
            return
        detector.process_video(source=args.source, output_file=args.output)
    elif args.mode == 'image':
        if not args.source:
            logger.error("Debes especificar --source para el modo image")
            return
        detections = detector.process_image(args.source, args.output)
        
        # Imprimir resultados
        for i, det in enumerate(detections, 1):
            status = "SÍ" if det['has_bottle'] else "NO"
            logger.info(f"Persona {i}: {status} tiene botella (confianza: {det['person_confidence']:.2f})")


if __name__ == "__main__":
    main()

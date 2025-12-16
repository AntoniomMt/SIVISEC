import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=10,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def calcular_orientacion_mano(landmarks, handedness):
    """
    Calcula la orientación de la mano de forma más precisa.
    Usa la posición relativa de los nudillos respecto a las puntas de los dedos.
    """
    # Puntos clave
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_mcp = landmarks[9]
    middle_pip = landmarks[10]
    pinky_tip = landmarks[20]
    pinky_mcp = landmarks[17]
    pinky_pip = landmarks[18]
    
    # Calcular si los dedos están "delante" o "detrás" de los nudillos en Z
    # Comparamos también con las articulaciones medias (PIP) para más precisión
    
    index_diff = index_tip.z - index_mcp.z
    middle_diff = middle_tip.z - middle_mcp.z
    pinky_diff = pinky_tip.z - pinky_mcp.z
    
    # También verificar las articulaciones medias vs los nudillos base
    index_pip_diff = index_pip.z - index_mcp.z
    middle_pip_diff = middle_pip.z - middle_mcp.z
    pinky_pip_diff = pinky_pip.z - pinky_mcp.z
    
    # Promedio de diferencias
    avg_diff = (index_diff + middle_diff + pinky_diff) / 3
    avg_pip_diff = (index_pip_diff + middle_pip_diff + pinky_pip_diff) / 3
    
    # Calcular ángulo de rotación usando las coordenadas X e Y
    # Vector de la palma (de muñeca al dedo medio)
    palm_x = middle_mcp.x - wrist.x
    palm_y = middle_mcp.y - wrist.y
    
    # Vector horizontal (índice a meñique)
    horizontal_x = index_mcp.x - pinky_mcp.x
    horizontal_y = index_mcp.y - pinky_mcp.y
    
    # Calcular si está muy rotada (mano de lado)
    h_length = np.sqrt(horizontal_x**2 + horizontal_y**2)
    v_length = np.sqrt(palm_x**2 + palm_y**2)
    
    # Si la mano está muy de lado, el vector horizontal es muy pequeño
    rotation_ratio = h_length / (v_length + 0.001)
    
    # Clasificar con lógica mejorada
    if rotation_ratio < 0.4:  # Mano muy de lado
        return 'rotacion'
    elif avg_diff < -0.015 and avg_pip_diff < -0.01:  
        # Tanto las puntas como las articulaciones medias están más cerca = PALMA
        return 'palma'
    else:  
        # En cualquier otro caso (incluido dorso con dedos extendidos) = DORSO
        return 'dorso'

def obtener_bbox(landmarks, img_width, img_height):
    """
    Calcula el bounding box de la mano detectada.
    """
    x_coords = [lm.x for lm in landmarks]
    y_coords = [lm.y for lm in landmarks]
    
    # Añadir margen
    margen = 0.05
    x_min = max(0, int((min(x_coords) - margen) * img_width))
    x_max = min(img_width, int((max(x_coords) + margen) * img_width))
    y_min = max(0, int((min(y_coords) - margen) * img_height))
    y_max = min(img_height, int((max(y_coords) + margen) * img_height))
    
    return x_min, y_min, x_max, y_max

def obtener_color_orientacion(orientacion):
    """
    Retorna el color BGR según la orientación.
    """
    colores = {
        'palma': (144, 238, 144),    # Verde claro
        'dorso': (255, 144, 30),     # Azul
        'rotacion': (0, 100, 0)      # Verde oscuro
    }
    return colores.get(orientacion, (255, 255, 255))

# Captura de video
cap = cv2.VideoCapture(0)

print("Iniciando detección de manos...")
print("- Verde claro: Palma")
print("- Azul: Dorso")
print("- Verde oscuro: Rotación")
print("Presiona 'q' para salir")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Voltear la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    img_height, img_width, _ = frame.shape
    
    # Convertir BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar la imagen
    results = hands.process(rgb_frame)
    
    # Detectar y dibujar manos
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Calcular orientación
            orientacion = calcular_orientacion_mano(
                hand_landmarks.landmark, 
                handedness.classification[0].label
            )
            color = obtener_color_orientacion(orientacion)
            
            # Obtener bounding box
            x_min, y_min, x_max, y_max = obtener_bbox(
                hand_landmarks.landmark, img_width, img_height
            )
            
            # Dibujar caja con color según orientación
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Etiqueta con el tipo de orientación
            label = orientacion.upper()
            cv2.putText(
                frame, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            
            # Dibujar landmarks para debugging (opcional)
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2)
            )
    
    # Mostrar leyenda
    cv2.putText(frame, "Verde claro: Palma", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (144, 238, 144), 2)
    cv2.putText(frame, "Azul: Dorso", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 144, 30), 2)
    cv2.putText(frame, "Verde oscuro: Rotacion", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 0), 2)
    
    # Mostrar resultado
    cv2.imshow('Deteccion de Manos - Orientacion', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
hands.close()

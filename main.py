from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from rembg import remove
from PIL import Image, ImageDraw, ImageOps
import mediapipe as mp
import numpy as np
import cv2
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_face_detection = mp.solutions.face_detection

@app.post("/procesar-avatar")
async def procesar_avatar(file: UploadFile = File(...)):
    contents = await file.read()
    # exif_transpose evita que las fotos de celular salgan de lado
    input_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGBA")

    # 1. Quitar fondo con la IA
    output_image = remove(input_image)
    W, H = output_image.size

    img_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGBA2RGB)
    
    mirar_izquierda = False
    face_found = False
    cx, cy, f_size = W // 2, H // 2, min(W, H)
    
    # 2. Buscar la cara
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4) as face_detection:
        results = face_detection.process(img_cv2)
        
        if results.detections:
            face_found = True
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Tamaño real de la cara en píxeles
            fw = int(bbox.width * W)
            fh = int(bbox.height * H)
            f_size = max(fw, fh)
            
            # Centro de la cara
            cx = int((bbox.xmin + bbox.width / 2) * W)
            cy = int((bbox.ymin + bbox.height / 2) * H)
            
            # LÓGICA DE EFECTO ESPEJO
            keypoints = detection.location_data.relative_keypoints
            right_ear = keypoints[4]
            left_ear = keypoints[5]
            nose = keypoints[2]
            
            dist_left = nose.x - right_ear.x
            dist_right = left_ear.x - nose.x
            
            if dist_left < (dist_right * 0.8):
                mirar_izquierda = True
                
    # 3. EL RECORTE ESTRICTO (Adiós al efecto pecera)
    if face_found:
        # El cuadrado de recorte será 2.5 veces el tamaño de tu cara
        S = int(f_size * 2.5) 
    else:
        S = min(W, H)

    # NUNCA permitimos que el cuadrado sea más grande que la foto
    S = min(S, W, H)

    # Calculamos la esquina superior izquierda del cuadrado
    x1 = cx - S // 2
    y1 = cy - int(S * 0.35) # La cara queda en el 35% superior del cuadro

    # 🚨 LA REGLA DE ORO: No salirse de los bordes 🚨
    # Si el cuadrado se sale por los lados, lo empujamos hacia adentro
    if x1 < 0:
        x1 = 0
    elif x1 + S > W:
        x1 = W - S
        
    # Si el cuadrado se sale por arriba o por abajo, lo empujamos
    if y1 < 0:
        y1 = 0
    elif y1 + S > H:
        y1 = H - S # Esto ancla el recorte a la base de tu foto

    # Recortamos exactamente ese cuadrado (ni un pixel transparente extra)
    cropped = output_image.crop((x1, y1, x1 + S, y1 + S))

    if mirar_izquierda:
        cropped = ImageOps.mirror(cropped)

    # Estiramos ese cuadrado perfecto al tamaño final
    CANVAS_SIZE = 600
    MARGIN = 10
    cropped = cropped.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)

    # 4. COMPOSICIÓN Y MÁSCARA
    # Fondo base azul
    bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    bg_draw.ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill="#9EAEE3")

    # Pegamos a la persona (como el cuadrado está justo al borde, llenará el círculo abajo)
    bg.paste(cropped, (0, 0), cropped)

    # Creamos un cuchillo circular para cortar los hombros que se salgan del círculo
    mask = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill=255)

    # Resultado final
    final_output = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    final_output.paste(bg, (0, 0), mask)

    # 5. Enviar a React
    img_byte_arr = io.BytesIO()
    final_output.save(img_byte_arr, format='PNG')
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")
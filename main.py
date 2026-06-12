import os
import io
import cv2
import numpy as np
import mediapipe as mp
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageOps
from rembg import remove

app = FastAPI()

# Permitimos tanto el dominio en producción como el entorno local por si necesitas depurar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.bracketzen.com", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar modelos al arrancar (fuera de la función para no consumir RAM en cada petición)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)

@app.post("/procesar-avatar")
async def procesar_avatar(file: UploadFile = File(...)):
    contents = await file.read()
    input_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGBA")

    # 1. Quitar fondo con IA
    output_image = remove(input_image)
    W, H = output_image.size

    img_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGBA2RGB)
    
    mirar_izquierda = False
    face_found = False
    cx, cy, f_size = W // 2, H // 2, min(W, H)
    
    # 2. Buscar la cara con el modelo pre-cargado
    results = face_detector.process(img_cv2)
    
    if results.detections:
        face_found = True
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        fw = int(bbox.width * W)
        fh = int(bbox.height * H)
        f_size = max(fw, fh)
        
        cx = int((bbox.xmin + bbox.width / 2) * W)
        cy = int((bbox.ymin + bbox.height / 2) * H)
        
        keypoints = detection.location_data.relative_keypoints
        right_ear = keypoints[4]
        left_ear = keypoints[5]
        nose = keypoints[2]
        
        dist_left = nose.x - right_ear.x
        dist_right = left_ear.x - nose.x
        
        if dist_left < (dist_right * 0.8):
            mirar_izquierda = True
            
    # 3. Recorte
    S = int(f_size * 2.5) if face_found else min(W, H)
    S = min(S, W, H)
    x1 = max(0, min(cx - S // 2, W - S))
    y1 = max(0, min(cy - int(S * 0.35), H - S))

    cropped = output_image.crop((x1, y1, x1 + S, y1 + S))

    if mirar_izquierda:
        cropped = ImageOps.mirror(cropped)

    CANVAS_SIZE = 600
    MARGIN = 10
    cropped = cropped.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)

    # 4. Composición Circular
    bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    bg_draw = ImageDraw.Draw(bg)
    bg_draw.ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill="#9EAEE3")
    bg.paste(cropped, (0, 0), cropped)

    mask = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill=255)

    final_output = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    final_output.paste(bg, (0, 0), mask)

    # 5. Respuesta
    img_byte_arr = io.BytesIO()
    final_output.save(img_byte_arr, format='PNG')
    
    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

# --- ARRANQUE ---
if __name__ == "__main__":
    import uvicorn
    # Forzamos que escuche en 0.0.0.0 y el puerto 10000
    uvicorn.run(app, host="0.0.0.0", port=10000)
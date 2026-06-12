import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import cv2
import numpy as np
import uvicorn
import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, ImageDraw, ImageOps
import mediapipe as mp
from rembg import remove, new_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.bracketzen.com", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización de IA
print("Cargando detector facial...")
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)

print("Cargando modelo de alta calidad (u2net)...")
rembg_session = new_session("u2net") 

@app.post("/procesar-avatar")
def procesar_avatar(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        input_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGBA")
        
        # Pre-escalado: Si la imagen es muy grande, redúcela para que la IA trabaje mejor y más rápido
        if max(input_image.size) > 1200:
            input_image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            
        W, H = input_image.size
        
        # Quitar fondo con modelo de alta calidad
        output_image = remove(input_image, session=rembg_session)
        
        # Detección de cara para encuadre
        img_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGBA2RGB)
        results = face_detector.process(img_cv2)
        
        cx, cy, S = W // 2, H // 2, min(W, H)
        mirar_izquierda = False

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Cálculo de centro y tamaño de zoom
            fw, fh = bbox.width * W, bbox.height * H
            f_size = max(fw, fh)
            cx = int((bbox.xmin + bbox.width / 2) * W)
            cy = int((bbox.ymin + bbox.height / 2) * H)
            
            # Lógica de encuadre: zoom 1.6 para mostrar cabeza y hombros
            S = int(f_size * 1.6)
            
            # Ajuste vertical: subimos el centro para que la cara no quede muy abajo
            y1 = max(0, min(cy - int(S * 0.45), H - S))
            x1 = max(0, min(cx - S // 2, W - S))
            
            # Detección de dirección
            keypoints = detection.location_data.relative_keypoints
            if keypoints[2].x - keypoints[4].x < (keypoints[5].x - keypoints[2].x) * 0.8:
                mirar_izquierda = True
        else:
            # Fallback si no hay cara: centrar imagen cuadrada
            x1, y1 = (W - S) // 2, (H - S) // 2

        # Recorte y aplicación de espejo
        cropped = output_image.crop((x1, y1, x1 + S, y1 + S))
        if mirar_izquierda:
            cropped = ImageOps.mirror(cropped)

        # Composición Circular (600x600)
        CANVAS_SIZE = 600
        MARGIN = 15 # Margen interno un poco mayor para estética
        cropped = cropped.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)
        
        bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg)
        bg_draw.ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill="#9EAEE3")
        bg.paste(cropped, (0, 0), cropped)

        # Máscara para bordes redondeados
        mask = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        ImageDraw.Draw(mask).ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill=255)

        final_output = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        final_output.paste(bg, (0, 0), mask)

        img_byte_arr = io.BytesIO()
        final_output.save(img_byte_arr, format='PNG')
        
        # Limpieza
        del input_image, output_image, cropped, bg, mask
        gc.collect()

        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        return Response(status_code=500, content=f"Error en procesar_avatar: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
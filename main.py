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
from rembg import remove, new_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.bracketzen.com", "http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cargar modelos
print("Cargando detector facial...")
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)

print("Cargando modelo ligero de recorte...")
light_session = new_session("u2netp") 

# 2. EL TRUCO MAESTRO: Calentar el motor. 
# Procesamos una imagen falsa de 10x10 para obligar a Render a descargar los archivos
# de la IA ahora, y no cuando un usuario suba su primera foto.
print("Calentando la IA...")
dummy_img = Image.new("RGBA", (10, 10), (255, 255, 255, 255))
_ = remove(dummy_img, session=light_session)
print("IA lista y esperando fotos.")

# 3. LA CORRECCIÓN DE FASTAPI: 'def' normal en lugar de 'async def'
@app.post("/procesar-avatar")
def procesar_avatar(file: UploadFile = File(...)):
    try:
        # Usamos file.file.read() porque ya no es una función asíncrona
        contents = file.file.read()
        input_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGBA")
        
        # Quitar fondo
        output_image = remove(input_image, session=light_session)
        W, H = output_image.size
        img_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGBA2RGB)
        
        # Detección de cara
        results = face_detector.process(img_cv2)
        face_found = False
        mirar_izquierda = False
        cx, cy, f_size = W // 2, H // 2, min(W, H)

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
            dist_left = keypoints[2].x - keypoints[4].x
            dist_right = keypoints[5].x - keypoints[2].x
            if dist_left < (dist_right * 0.8):
                mirar_izquierda = True
                
        # Recorte Estricto
        S = min(int(f_size * 2.5) if face_found else min(W, H), W, H)
        x1 = max(0, min(cx - S // 2, W - S))
        y1 = max(0, min(cy - int(S * 0.35), H - S))

        cropped = output_image.crop((x1, y1, x1 + S, y1 + S))
        if mirar_izquierda:
            cropped = ImageOps.mirror(cropped)

        # Composición circular
        CANVAS_SIZE = 600
        MARGIN = 10
        cropped = cropped.resize((CANVAS_SIZE, CANVAS_SIZE), Image.Resampling.LANCZOS)
        
        bg = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(bg)
        bg_draw.ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill="#9EAEE3")
        bg.paste(cropped, (0, 0), cropped)

        mask = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        ImageDraw.Draw(mask).ellipse((MARGIN, MARGIN, CANVAS_SIZE - MARGIN, CANVAS_SIZE - MARGIN), fill=255)

        final_output = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
        final_output.paste(bg, (0, 0), mask)

        img_byte_arr = io.BytesIO()
        final_output.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"Error crítico en el backend: {e}")
        return Response(status_code=500, content=f"Error interno: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
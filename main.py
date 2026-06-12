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

# 🚨 EL CAMBIO CRÍTICO: Cargar la IA Ligera 🚨
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)

# Usamos 'u2netp' (Piccolo) en lugar del modelo pesado por defecto para evitar el 502
light_session = new_session("u2netp") 

@app.post("/procesar-avatar")
async def procesar_avatar(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        input_image = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGBA")
        
        # 1. Quitar fondo pasándole la sesión ligera
        output_image = remove(input_image, session=light_session)
        
        W, H = output_image.size
        img_cv2 = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGBA2RGB)
        
        # 2. Detección de cara
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
                
        # 3. Recorte Estricto
        S = min(int(f_size * 2.5) if face_found else min(W, H), W, H)
        x1 = max(0, min(cx - S // 2, W - S))
        y1 = max(0, min(cy - int(S * 0.35), H - S))

        cropped = output_image.crop((x1, y1, x1 + S, y1 + S))
        if mirar_izquierda:
            cropped = ImageOps.mirror(cropped)

        # 4. Composición
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
        print(f"Error procesando la imagen: {e}")
        return Response(status_code=500, content="Error interno procesando el avatar")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
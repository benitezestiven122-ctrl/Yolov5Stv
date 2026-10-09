from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

# ==========================================
# 1. CONFIGURACIÓN Y ESTILO VJ / DARK MODE
# ==========================================
st.set_page_config(
    page_title="Visio-Synth | Generador de Triggers",
    page_icon="🎛️",
    layout="wide"
)

# Estilo para darle un look de software de VJ/Producción
st.markdown("""
<style>
    .trigger-box {
        background-color: #1E1E1E;
        border-left: 5px solid #00FFAA;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
        font-family: monospace;
    }
    .highlight { color: #00FFAA; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# ==========================================
# 2. DICCIONARIO DE MAPEO CREATIVO
# ==========================================
# Asignamos efectos visuales o de sonido dependiendo del objeto detectado
def mapear_efecto(objeto):
    mapeo = {
        "person": {"capa": "Capa 1: Silueta de Partículas", "parametro": "Opacidad / Oscilador"},
        "cell phone": {"capa": "Capa 2: Efecto Glitch / Ruido", "parametro": "Intensidad de Distorsión"},
        "cup": {"capa": "Capa 3: Generador de Geometría 3D", "parametro": "Rotación X/Y"},
        "bottle": {"capa": "Capa 4: Modulador de Frecuencia", "parametro": "Filtro Pasa-bajos (Audio)"},
        "laptop": {"capa": "Capa 5: Feedback Loop", "parametro": "Escala / Zoom"},
    }
    return mapeo.get(objeto.lower(), {"capa": f"Generativo Base ({objeto})", "parametro": "Velocidad LFO"})

# ==========================================
# 3. INTERFAZ PRINCIPAL
# ==========================================
st.title("🎛️ Visio-Synth: Mapeador de Triggers por Visión")
st.markdown("Convierte objetos físicos en parámetros de control para software audiovisual. Usa la cámara para detectar elementos y generar mapeos de datos (Data Mapping) ideales para integraciones OSC/MIDI.")

with st.spinner("Inicializando motor de visión..."):
    model = load_model()

if model:
    with st.sidebar:
        st.header("🎚️ Parámetros del Sensor")
        st.caption("Ajusta la sensibilidad para evitar falsos triggers en vivo.")
        conf_threshold = st.slider("Umbral de Confianza", 0.0, 1.0, 0.40, 0.01)
        iou_threshold  = st.slider("Filtro de Superposición (IoU)", 0.0, 1.0, 0.45, 0.01)
        
        st.divider()
        st.write("📡 **Destinos de Salida (Simulados)**")
        st.checkbox("Enviar a TouchDesigner (OSC)", value=True)
        st.checkbox("Enviar a Resolume (MIDI)", value=True)

    picture = st.camera_input("Capturar escena / Input de video", key="camera")

    if picture:
        bytes_data = picture.getvalue()

        # Decodificación de imagen optimizada
        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR para YOLO

        with st.spinner("Procesando fotograma..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=50
                )
            except Exception as e:
                st.error(f"Error en el motor de render: {str(e)}")
                st.stop()

        result    = results[0]
        boxes     = result.boxes
        annotated = result.plot()              
        annotated_rgb = annotated[:, :, ::-1]  

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.subheader("👁️ Visión Artificial (Input)")
            st.image(annotated_rgb, use_container_width=True)

        with col2:
            st.subheader("📟 Panel de Triggers Activos")
            
            if boxes is not None and len(boxes) > 0:
                label_names = model.names
                objetos_detectados = set() # Usamos set para no repetir efectos si hay 2 personas

                for box in boxes:
                    cat = int(box.cls.item())
                    nombre_objeto = label_names[cat]
                    objetos_detectados.add(nombre_objeto)

                # Generar los bloques de la interfaz para cada efecto activado
                for obj in objetos_detectados:
                    efecto = mapear_efecto(obj)
                    st.markdown(f"""
                    <div class="trigger-box">
                        <span style="color: #888;">Objeto Detectado:</span> {obj.upper()}<br>
                        <span class="highlight">▶ ACTIVANDO:</span> {efecto['capa']}<br>
                        <span style="color: #888;">Modulando:</span> {efecto['parametro']}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Tabla de datos para visualización analítica
                st.write("📊 **Matriz de Datos a Enviar (JSON/OSC)**")
                data_export = [{"Objeto": obj, "Capa Asignada": mapear_efecto(obj)['capa']} for obj in objetos_detectados]
                st.dataframe(pd.DataFrame(data_export), use_container_width=True)
                
            else:
                st.info("Esperando estímulos visuales... Ningún trigger activado.")
else:
    st.error("Error crítico de sistema. Reinicie el entorno.")

st.markdown("---")
st.caption("🚀 **Arquitectura**: YOLOv5 (Detección) + Streamlit (UI). Diseñado para interactuar conceptualmente con entornos de programación visual y VJing.")

# Cruces Inteligentes con Edge AI

Proyecto de prototipo para un sistema de **cruce inteligente** basado en **visión por computadora en el borde (Edge AI)** usando:

- Raspberry Pi 4
- Dos cámaras USB
- Modelo **YOLOv5n en formato ONNX** con **OpenCV DNN**
- Imagen Linux mínima generada con **Yocto Project**
- Control de semáforo por máquina de estados en Python
- Comunicación entre procesos mediante archivos en `/tmp`

El sistema detecta peatones, vehículos y animales, y ajusta el semáforo según reglas locales orientadas a la **seguridad**.

---

## 6. Vista Funcional del Sistema

El sistema se organiza en varios módulos funcionales que cooperan entre sí para capturar video, detectar actores viales, tomar decisiones y controlar el semáforo.

### 6.1 Módulos principales

1. **Módulo de captura de video (por proceso)**
   - Cada script (`people_counter_cam.py` y `veh_counter_cam.py`) abre una cámara USB independiente.
   - Realiza:
     - Captura continua de frames con V4L2.
     - Redimensionamiento a la resolución de trabajo (p. ej. 960×540).
     - Conversión de color y normalización de píxeles para el modelo ONNX.

2. **Detección y clasificación de objetos (YOLOv5n ONNX + OpenCV DNN)**
   - Ambos scripts usan un modelo **YOLOv5n** exportado a **ONNX**, ejecutado con `cv2.dnn`.
   - Funciones:
     - Detección de **personas** en el módulo peatonal.
     - Detección de **vehículos** y **animales** en el módulo vehicular.
     - Filtrado por confianza mínima y Non-Maximum Suppression (NMS).
   - El módulo vehicular incluye además:
     - Región de interés opcional mediante máscara.
     - Separación explícita entre clases de vehículo y clases de animal.

3. **Seguimiento y análisis de movimiento (VehCam)**
   - El script `veh_counter_cam.py` integra:
     - **Flujo óptico (FlowGate)** para distinguir objetos realmente en movimiento.
     - **SmoothTracker (Kalman + EMA)** para:
       - Asociar detecciones a lo largo del tiempo.
       - Contabilizar vehículos en movimiento de forma robusta.
   - Esto permite generar un conteo de “vehículos en movimiento” menos sensible a ruido.

4. **Lógica de flags y salida por proceso**
   - Cada módulo produce su propia salida en formato JSON por consola y en archivos en `/tmp`:
     - `people_counter_cam.py`:
       - `/tmp/ped_count.txt`: número de personas detectadas.
       - `/tmp/ped_flag.txt`: flag binario si el conteo supera un umbral.
     - `veh_counter_cam.py`:
       - `/tmp/veh_moving.txt`: conteo de vehículos en movimiento.
       - `/tmp/veh_flag.txt`: flag binario de presencia vehicular relevante.
       - `/tmp/animal_flag.txt`: flag persistente de animal detectado (con cooldown de limpieza).
   - Estos archivos son la interfaz de comunicación hacia el controlador del semáforo.

5. **Controlador del semáforo (máquina de estados)**
   - El script `traffic_control.py` (controlador principal) lee periódicamente:
     - `/tmp/ped_flag.txt`
     - `/tmp/veh_flag.txt`
     - `/tmp/animal_flag.txt`
   - Aplica una máquina de estados que controla:
     - Fases de **verde/rojo para vehículos**.
     - Fase **peatonal**.
     - **Bloqueo por animal** (estado de seguridad).
     - **Cooldown de seguridad** tras cambios de fase.
   - El resultado se registra en logs con marcas de tiempo, reflejando:
     - Estados de las luces.
     - Motivo de cada transición (peatones, animal, ciclo normal, cooldown, etc.).

6. **Interfaz gráfica opcional (GUI)**
   - `traffic_gui.py` es una interfaz Tkinter para:
     - Lanzar y detener los procesos durante desarrollo.
     - Ver en vivo los logs de cada módulo.
     - Mostrar el estado del semáforo en una ventana de escritorio.
   - Esta GUI se emplea en entorno de desarrollo (PC / Raspbian) y no está integrada en la imagen final de Yocto, pero puede añadirse como trabajo futuro.

---

## 7. Arquitectura del Sistema Propuesto (Hardware y Software)

### 7.1 Arquitectura de Hardware

El prototipo se implementa sobre un nodo embebido basado en:

- **Raspberry Pi 4 Model B (4 GB RAM)**
  - CPU ARM quad-core, suficiente para ejecutar YOLOv5n ONNX en CPU en tiempo real (con resoluciones moderadas).
- **Dos cámaras USB UVC**
  - Una para el módulo peatonal (**PeatonCam**).
  - Otra para el módulo vehicular (**VehCam**).
  - Conectadas por puertos USB y accedidas vía V4L2 (`/dev/video*`).
- **Almacenamiento**
  - Tarjeta microSD con la imagen de Yocto generada específicamente para el proyecto.
- **Alimentación**
  - Fuente estándar para Raspberry Pi 4 (5V, 3A).
- **Conectividad (opcional)**
  - Ethernet / WiFi integrados.
  - En el prototipo, la conectividad se usó principalmente para depuración por SSH; no se implementó comunicación con un servidor central.

> Nota: Aunque el sistema está preparado conceptualmente para controlar un semáforo físico mediante GPIO + relés, el prototipo se enfoca en la lógica de control y el registro de estados. La conexión a hardware real puede añadirse como extensión.

---

### 7.2 Arquitectura de Software

La arquitectura de software se organiza en capas:

#### Capa 1 – Sistema Operativo Embebido (Yocto)

- Imagen Linux mínima generada con **Yocto Project** para Raspberry Pi 4.
- Soporte para:
  - Python 3
  - OpenCV 4.5.5 (incluyendo módulo DNN)
  - V4L2 (cámaras USB)
  - systemd
- Servicio `traffic-app.service` que:
  - Invoca el script `/opt/traffic-app/start-traffic-app.sh` al arranque.
  - Inicia los tres procesos principales (PeatonCam, VehCam y controlador).

#### Capa 2 – Aplicación principal (scripts en `/opt/traffic-app/`)

- `people_counter_cam.py`
  - Abre la cámara peatonal.
  - Carga `yolov5n.onnx` con OpenCV DNN.
  - Detecta personas y genera conteos y flags.

- `veh_counter_cam.py`
  - Abre la cámara vehicular.
  - Carga `yolov5n.onnx` con OpenCV DNN.
  - Distingue vehículos y animales.
  - Aplica flujo óptico + tracker para vehículos en movimiento.
  - Genera flags de vehículo y de animal.

- `traffic_control.py`
  - Implementa la máquina de estados del semáforo.
  - Lee las flags en `/tmp`.
  - Registra la evolución del sistema en logs estilo timeline.

- `traffic_gui.py` (uso opcional en desarrollo)
  - Interfaz gráfica para PC / entorno Raspbian.
  - No parte de la imagen Yocto final.

- `yolov5n.onnx`
  - Modelo de detección optimizado para CPU.
  - Seleccionado específicamente para ser compatible con OpenCV 4.5.5 del entorno Yocto.

#### Capa 3 – Comunicación entre módulos

- Todos los procesos se coordinan mediante **archivos de texto** en `/tmp`:
  - `ped_count.txt`, `ped_flag.txt`
  - `veh_moving.txt`, `veh_flag.txt`
  - `animal_flag.txt`
- Ventajas:
  - Sencillo de depurar.
  - No requiere sockets ni colas de mensajes.
  - Adecuado para un prototipo embebido monolítico.

---

## 8. Dependencias de Software

Las dependencias reflejan la implementación real del prototipo, tanto en desarrollo como en la imagen Yocto.

### 8.1 Dependencias principales (aplicación)

- **Python 3**
  - Lenguaje base de todos los scripts.

- **OpenCV 4.5.5 (con módulo DNN)**
  - Captura de video (VideoCapture).
  - Preprocesamiento (resize, conversión de color, blobs).
  - Inferencia sobre `yolov5n.onnx` mediante `cv2.dnn.readNetFromONNX`.
  - Operaciones auxiliares (dibujado de cajas en modo debug/GUI).

- **NumPy**
  - Manipulación de tensores y arreglos numéricos.
  - Operaciones sobre salidas del modelo y flujo óptico.

- **Standard Library de Python**
  - `argparse`: manejo de argumentos por línea de comandos.
  - `time`, `json`, `os`, `math`: utilidades generales.
  - `subprocess`, `threading`, `queue` (para la GUI).

- **Tkinter** (solo para GUI opcional)
  - Implementación de `traffic_gui.py` en entorno de desarrollo.

### 8.2 Dependencias del sistema embebido (Yocto / Linux)

- **Yocto Project + meta-raspberrypi**
  - Generación de la imagen Linux mínima para Raspberry Pi 4.
- **systemd**
  - Manejo del servicio `traffic-app.service`.
- **BusyBox / coreutils**
  - Herramientas básicas de sistema.
- **V4L2 y controladores UVC**
  - Soporte para cámaras USB.
- **GStreamer (soporte básico vía OpenCV)**
  - Backend de captura en algunos pipelines de OpenCV.

> Importante: **TensorFlow Lite y YOLOv8 no se utilizaron en el prototipo final.**  
> Se reemplazaron por **YOLOv5n ONNX + OpenCV DNN** debido a compatibilidad y peso de las dependencias en Yocto.

---

## 9. Estrategia de Integración y Despliegue

La integración del sistema se realizó de forma incremental, combinando desarrollo en PC, pruebas en Raspbian y despliegue final en Yocto.

### 9.1 Fases de integración

1. **Diseño inicial y propuesta**
   - Definición de la idea de cruces inteligentes con Edge AI.
   - Redacción de la propuesta técnica y arquitectónica.

2. **Prototipo de control de semáforo en Python**
   - Implementación de una máquina de estados para:
     - Fases vehiculares.
     - Fase peatonal.
     - Estados especiales por detección de animal.
   - El controlador se pensó desde el inicio para consumir flags de detección externos.

3. **Pruebas de detección en entorno de escritorio**
   - Ejecución de scripts de detección en PC con GPU / CPU.
   - Uso inicial de modelos orientados a GPU, que luego resultaron incompatibles con el entorno Yocto de la Raspberry Pi.

4. **Migración a modelo compatible (YOLOv5n ONNX + OpenCV DNN)**
   - Identificación de incompatibilidades entre ciertas variantes de YOLO y OpenCV 4.5.5.
   - Selección y prueba de `yolov5n.onnx`, validando:
     - Carga correcta en la Raspberry Pi.
     - Formato de salida `(25200, 85)`.
   - Ajuste del postprocesamiento:
     - Corrección del escalado de coordenadas.
     - Adaptación del orden de ejes y del formato de detecciones.

5. **Separación en procesos y comunicación por `/tmp`**
   - Definición de tres procesos:
     - PeatonCam (peatones).
     - VehCam (vehículos + animales).
     - Controlador de semáforo.
   - Establecimiento de un protocolo basado en archivos de texto:
     - Flags y conteos actualizados frame a frame.
     - Lectura periódica desde el controlador.

6. **Construcción de la imagen Yocto**
   - Creación de una imagen mínima con:
     - Python + OpenCV + dependencias básicas.
   - Adición de:
     - Scripts de aplicación en `/opt/traffic-app/`.
     - Modelo `yolov5n.onnx`.
     - Script de arranque `start-traffic-app.sh`.
     - Servicio `traffic-app.service`.

7. **Depuración en la Raspberry Pi**
   - Revisión de errores de arranque (problemas de flasheo).
   - Comprobación de que:
     - El servicio se ejecuta correctamente.
     - Las cámaras se abren sin problemas.
     - Los scripts escriben correctamente en `/tmp`.
   - Ajustes finales vía SSH:
     - Resolución de la cámara.
     - Frecuencia de detección.
     - Afinado de flags de animal y de vehículos.

8. **Integración de GUI (en entorno de desarrollo)**
   - Implementación de `traffic_gui.py` para:
     - Visualizar logs de los procesos.
     - Ver en vivo el estado del semáforo.
   - Por razones de tiempo y estabilidad de la imagen, la GUI no se integró en la imagen Yocto final, pero se conserva como herramienta adicional de demostración.

---

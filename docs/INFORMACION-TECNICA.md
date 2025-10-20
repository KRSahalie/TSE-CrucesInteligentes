# 6. Vista Funcional del Sistema: Descomposición de Funciones

El sistema **Cruces Inteligentes con Edge AI** se descompone en módulos funcionales esenciales para cumplir con los requerimientos establecidos anteriormente. Esta vista describe las capacidades del sistema, basándose en la necesidad de captura, procesamiento y clasificación de objetos en cruces concurridos, además de los módulos para el control del semáforo del cruce.

### I. Captura y Preprocesamiento de Video
Este módulo se encarga de adquirir el flujo de video en tiempo real desde las cámaras conectadas al Raspberry Pi. Los frames obtenidos se preprocesan para optimizar el rendimiento del modelo de IA mediante técnicas como redimensionamiento, filtrado y normalización. Además, se segmenta el entorno del cruce para diferenciar vías vehiculares, pasos peatonales y zonas de fauna, facilitando la clasificación contextual de los objetos detectados.

### II. Procesamiento e Inferencia con IA
Aquí se ejecutan los modelos de aprendizaje automático sobre los frames preprocesados. El sistema detecta y clasifica en tiempo real vehículos, peatones y fauna, analizando su movimiento para estimar dirección y velocidad. Con esta información, el módulo toma decisiones locales sobre el estado del cruce, como activar o no el paso peatonal, priorizando la seguridad y la fluidez del tránsito.

### III. Control del Cruce Inteligente
Este módulo traduce las decisiones de la IA en acciones concretas sobre el semáforo. En función del análisis del entorno, activa o desactiva las luces (rojo, verde) en la simulación. La visualización del semáforo puede implementarse en Python utilizando bibliotecas como OpenCV, Tkinter o Pygame, permitiendo mostrar de manera gráfica el estado de las señales. Además, este módulo gestiona prioridades de paso según la densidad de tráfico y supervisa que los actuadores respondan correctamente, aunque el semáforo sea virtual.

### IV. Sistema Embebido y Software Base
Se encarga de la administración del entorno de ejecución sobre la Raspberry Pi. Incluye la construcción de la imagen de Linux embebido con Yocto Project, integrando las dependencias necesarias (OpenCV, TensorFlow Lite). Además, gestiona los recursos del sistema y asegura la inicialización automática y el monitoreo constante del servicio principal que mantiene funcionando todo el sistema.

### V. Registro y Monitoreo del Sistema
Este módulo documenta la operación del sistema para su validación y depuración. Registra eventos importantes, como detecciones y decisiones, permite la visualización del estado del cruce mediante una interfaz local o remota, y facilita la exportación de datos a una red de monitoreo o almacenamiento local para análisis posterior.

| Módulo Funcional | Descripción General | Funciones Específicas Requeridas |
|-----------------|------------------|--------------------------------|
| I. Captura y Preprocesamiento de Video | Control del flujo de video proveniente de la cámara conectada al Raspberry Pi, preparando las imágenes para análisis de IA. | 1. Captura de video en tiempo real. 2. Preprocesamiento (redimensionar, filtrar, normalizar). 3. Segmentación del entorno (calles, pasos peatonales, zonas de fauna). |
| II. Procesamiento e Inferencia con IA | Uso de modelos de aprendizaje automático (TensorFlow Lite + OpenCV) para detectar entidades relevantes. | 1. Detección y clasificación de vehículos, peatones y fauna. 2. Análisis de movimiento (dirección y velocidad). 3. Toma de decisiones local para paso peatonal. |
| III. Control del Cruce Inteligente | Gestiona señales del semáforo según el procesamiento de IA. | 1. Activación de señales (rojo/verde). 2. Gestión de prioridades según densidad de tráfico. 3. Supervisión del estado de los actuadores. |
| IV. Sistema Embebido y Software Base | Administración del entorno de ejecución sobre Raspberry Pi, integración del SO embebido, bibliotecas y servicios. | 1. Integración Yocto Project (OpenCV, TensorFlow Lite, GPIO). 2. Gestión de recursos (CPU, memoria, sensores). 3. Inicialización y monitoreo del servicio principal. |
| V. Registro y Monitoreo del Sistema | Registro de eventos y resultados para validación y depuración. | 1. Registro de actividad (detecciones, decisiones). 2. Interfaz de diagnóstico local o remota. 3. Exportación de datos a red o almacenamiento local. |

# 7. Arquitectura del Sistema Propuesto (Hardware y Software)

La arquitectura define la estructura física y lógica del nodo de monitoreo, implementando un esquema de **Edge Computing** donde el procesamiento ocurre en el dispositivo, no en la nube.  
El nodo procesa video en tiempo real para la detección de peatones, vehículos y fauna, y controla un semáforo virtual mostrando el flujo del cruce en una pantalla conectada al sistema.

## Arquitectura de Hardware 💾

El nodo se basa en una plataforma de bajo costo y alta flexibilidad:

- **Plataforma de Cómputo Embebido:** Raspberry Pi 4 Model B (4GB RAM), adecuada para procesamiento de video en tiempo real y ejecución de modelos TensorFlow Lite.  
- **Periféricos de Entrada (Visión):** Dos cámaras USB conectadas a la Raspberry Pi, encargadas de capturar video de alta resolución de diferentes ángulos del cruce. Se usan puertos USB 3.0 para asegurar ancho de banda suficiente.  
- **Periférico de Salida (Visualización):** Pantalla HDMI conectada a la Raspberry Pi para mostrar el flujo de video en tiempo real, el estado del semáforo virtual y métricas relevantes.  
- **Almacenamiento:** Tarjeta MicroSD Kingston Canvas Select Plus de 32GB, suficiente para almacenar el sistema operativo, modelos de IA y datos temporales de prueba.  
- **Conectividad:** Módulo Ethernet/WiFi integrado para la comunicación de métricas y gestión remota del nodo dentro de la red de monitoreo.

## Arquitectura de Software 💻

La estructura de software se organiza en cuatro capas principales:

### 1. Capa de Aplicación (Edge AI)
- **Algoritmo de Monitoreo:** Código que orquesta la captura, detección, clasificación y tracking de objetos en tiempo real desde las cámaras.  
- **Modelos de ML:** Archivos optimizados (`.tflite`) que ejecutan la inferencia directamente en la Raspberry Pi.

### 2. Capa de Control del Semáforo y Visualización
- **Simulación de Semáforo:** Visualización del estado de las señales (rojo/amarillo/verde) utilizando Python con bibliotecas como OpenCV, Tkinter o Pygame.  
- **Gestión de Prioridades:** Ajuste de tiempos de cambio de luz según la densidad de tráfico y presencia de peatones.  
- **Visualización en Pantalla:** Muestra el flujo de video en tiempo real, el semáforo virtual y métricas importantes para supervisión.  
- **Supervisión de Actuadores:** Asegura que los cambios de estado se apliquen correctamente.

### 3. Capa de Middleware (Librerías)
- **TensorFlow Lite:** Runtime eficiente para la ejecución de modelos de IA en la Raspberry Pi.  
- **OpenCV:** Librería de Visión por Computador para preprocesamiento de frames, segmentación de zonas del cruce y utilidades de imagen.

### 4. Capa de Sistema Operativo
- **Linux Embebido:** Imagen mínima optimizada para Raspberry Pi 4, generada mediante Yocto Project.  
- **Kernel Linux:** Núcleo configurado para optimizar el rendimiento, la gestión de periféricos y la comunicación con cámaras USB.

# 8. Dependencias de Software 📦

| Dependencia | Tipo | Propósito |
|-------------|------|-----------|
| **TensorFlow Lite (TFLite)** | Runtime de ML | Ejecución de modelos de detección y clasificación en el Edge. |
| **OpenCV** | Librería de Visión | Manipulación de video, preprocesamiento de imágenes para el modelo de ML. |
| **Python 3** | Intérprete/Librerías | Lenguaje base para el desarrollo del código de la aplicación (si aplica). |
| **Drivers de Cámara (V4L2)** | Kernel/Drivers | Interfaz para la comunicación con el módulo de cámara de la Raspberry Pi. |

# 9. Estrategia de Integración (Yocto Project) 🛠️

La estrategia de integración asegura que la aplicación de **Edge AI** corra de manera robusta sobre un sistema operativo optimizado y personalizado para la **Raspberry Pi 4 Model B**. Se busca empaquetar todo en una **imagen única**, lista para deploy y ejecución en el nodo de monitoreo.

### 1. Identificación de Recetas
- Analizar las dependencias necesarias para el proyecto, incluyendo TensorFlow Lite, OpenCV y librerías adicionales como numpy, libjpeg y zlib.
- Identificar o sintetizar las recetas (`.bb files`) que permitan compilar estas dependencias para la arquitectura ARM de la Raspberry Pi.

### 2. Generación de Imagen Base
- Configurar Yocto Project para el target **Raspberry Pi 4 B**.
- Generar una imagen mínima de Linux embebido con soporte para CPU, GPU, USB 3.0, HDMI y WiFi/Ethernet.

### 3. Inclusión de Dependencias
- Modificar la receta de la imagen final para incluir todas las librerías y frameworks identificados.
- Asegurar la compilación correcta para ARM y realizar pruebas de compatibilidad.

### 4. Integración de la Aplicación Principal
- Crear la receta para la aplicación de monitoreo del cruce inteligente.
- Configurar la aplicación para que se ejecute automáticamente al iniciar el sistema operativo y controle las cámaras, semáforo virtual y envío de métricas.

### 5. Síntesis de Imagen Final
- Ejecutar `bitbake` para compilar la imagen final, que incluirá Linux embebido, librerías y la aplicación de Edge AI.
- Validar que la imagen generada sea funcional y estable.

### 6. Despliegue y Verificación
- Instalar la imagen en la MicroSD de la Raspberry Pi.
- Conectar las cámaras, la pantalla HDMI y verificar la operación completa del nodo (procesamiento de video en tiempo real, visualización del semáforo virtual, registro de métricas).

**Notas adicionales:**  
- Aún en desarrollo: se agregarán imágenes de la arquitectura y conexiones físicas.  
- Se ampliará la especificación de hardware incluyendo pantalla, cámaras, microSD y conectividad, para documentar completamente el nodo de monitoreo.


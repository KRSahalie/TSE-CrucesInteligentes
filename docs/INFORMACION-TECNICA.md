#La información técnica explica los puntos 6,7,8 y 9 de la entrega preliminar.

# 6. Vista Funcional del Sistema: Descomposición de Funciones

El sistema **Cruces Inteligentes con Edge AI** se descompone en módulos funcionales esenciales para cumplir su objetivo de monitoreo de tránsito. Esta vista describe las capacidades del sistema, basándose en la necesidad de captura, procesamiento y clasificación de objetos en cruces concurridos.

| Módulo Funcional | Descripción General | Funciones Específicas Requeridas |
|-----------------|------------------|--------------------------------|
| **I. Captura y Preprocesamiento** | Gestión del flujo de video de la cámara y preparación de los frames para la inferencia de IA. | **1.1. Captura de Video:** Adquirir el flujo de imágenes en tiempo real desde la cámara periférica.<br>**1.2. Preprocesamiento:** Aplicar optimizaciones (ej. redimensionamiento, normalización) para adecuar la imagen al modelo de IA y la capacidad de cómputo de la Raspberry Pi. |
| **II. Procesamiento Edge AI** | Ejecución de los algoritmos de aprendizaje máquina (TensorFlow Lite) para identificar, clasificar y rastrear elementos de tránsito. | **2.1. Detección y Clasificación:** Identificar objetos de interés (vehículos, peatones, ciclistas, fauna) mediante algoritmos de visión por computador y clasificarlos.<br>**2.2. Rastreo de Comportamiento:** Mantener el seguimiento (tracking) de los objetos detectados a lo largo de múltiples frames para analizar su trayectoria y velocidad. |
| **III. Gestión y Comunicación de Datos** | Generación de información de valor a partir de la inferencia de IA y su transmisión a la red de monitoreo central. | **3.1. Generación de Métricas:** Calcular conteos de objetos, densidad de tráfico o tiempos de cruce a partir de los datos rastreados.<br>**3.2. Comunicación de Alertas/Datos:** Transmitir los resultados y métricas del análisis local (Edge) a un sistema central a través de la red. |
| **IV. Operación del Sistema Embebido** | Gestión de la capa de sistema operativo y los recursos de la Raspberry Pi. | **4.1. Configuración de SO:** Asegurar la correcta inicialización del Linux Embebido (generado con Yocto) y la gestión eficiente de recursos de hardware.<br>**4.2. Administración de Aplicación:** Capacidad de iniciar, detener y monitorear el estado de la aplicación de Edge AI. |


# 7. Arquitectura del Sistema Propuesto (Hardware y Software)

La arquitectura define la estructura física y lógica del nodo de monitoreo, implementando un esquema de **Edge Computing** donde el procesamiento ocurre en el dispositivo, no en la nube.

---

## Arquitectura de Hardware 💾

El nodo se basa en una plataforma de bajo costo y alta flexibilidad:

- **Plataforma de Cómputo Embebido:** Raspberry Pi (modelo a especificar, preferiblemente uno reciente con capacidad de aceleración para IA).  
- **Periférico de Entrada (Visión):** Módulo de cámara conectado a la Raspberry Pi, encargado de la captura de video de alta resolución.  
- **Almacenamiento:** Tarjeta MicroSD para almacenar el Sistema Operativo personalizado y los modelos de TensorFlow Lite.  
- **Conectividad:** Módulo Ethernet/WiFi integrado para la comunicación de métricas y gestión remota del nodo dentro de la red de monitoreo.  

---

## Arquitectura de Software 💻

La estructura de software se organiza en tres capas principales:

### 1. Capa de Aplicación (Edge AI)
Contiene la lógica principal del sistema:

- **Algoritmo de Monitoreo:** Código ejecutable que orquesta la detección, clasificación y tracking.  
- **Modelos de ML:** Archivos optimizados (`.tflite`) que ejecutan la inferencia.  

### 2. Capa de Middleware (Librerías)
Proporciona las herramientas necesarias para la ejecución en el target:

- **TensorFlow Lite:** Runtime esencial para la ejecución eficiente del modelo de IA en la Raspberry Pi.  
- **OpenCV:** Librería de Visión por Computador para la gestión de video, preprocesamiento y utilidades de imagen.  

### 3. Capa de Sistema Operativo
La base del sistema:

- **Linux Embebido:** Imagen mínima y optimizada para el hardware de la Raspberry Pi, generada a través del flujo de trabajo de Yocto Project.  
- **Kernel Linux:** Núcleo configurado para optimizar el rendimiento y la gestión de periféricos.  


# 8. Dependencias de Software y 9. Estrategia de Integración

---

## Dependencias de Software 📦

Las dependencias son importantes para el desarrollo de la imagen con **Yocto Project**, ya que deben incluirse como recetas en el build system.

| Dependencia | Tipo | Propósito |
|-------------|------|-----------|
| **TensorFlow Lite (TFLite)** | Runtime de ML | Ejecución de modelos de detección y clasificación en el Edge. |
| **OpenCV** | Librería de Visión | Manipulación de video, preprocesamiento de imágenes para el modelo de ML. |
| **Python 3** | Intérprete/Librerías | Lenguaje base para el desarrollo del código de la aplicación (si aplica). |
| **Drivers de Cámara (V4L2)** | Kernel/Drivers | Interfaz para la comunicación con el módulo de cámara de la Raspberry Pi. |

---

## 9. Estrategia de Integración (Yocto Project) 🛠️

La estrategia garantiza que la aplicación de **Edge AI** corra de manera robusta sobre un sistema operativo optimizado:

1. **Identificación de Recetas:**  
   Analizar las dependencias listadas (TFLite, OpenCV) e identificar o sintetizar las recetas (`.bb files`) necesarias para incluirlas en el build de Yocto Project.

2. **Generación de Imagen Base:**  
   Configurar el flujo de Yocto para el target Raspberry Pi, y generar una imagen base mínima de Linux.

3. **Inclusión de Dependencias:**  
   Modificar la receta de la imagen final para incluir las recetas de TFLite, OpenCV y sus dependencias de librerías cruzadas.

4. **Integración de la Aplicación:**  
   Sintetizar la receta para la aplicación principal de monitoreo, asegurando que se compile y ejecute automáticamente al iniciar el sistema operativo.

5. **Síntesis de Imagen Final:**  
   Compilar la imagen de Linux personalizada con `bitbake`, que contendrá el sistema operativo, las dependencias y la aplicación de Edge AI en un único paquete para el deployment.

6. **Despliegue y Verificación:**  
   Instalar la imagen generada en la Raspberry Pi y demostrar su operación correcta.
   
   
   Aún en desarrollo, falta agregar imagenes de la arquitectura y sus conexiones. Ampliar y especificar el hardware etc.


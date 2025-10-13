# 🚦 Proyecto 2 – Cruces Inteligentes con Edge AI  

[![Estado](https://img.shields.io/badge/estado-en_desarrollo-blue.svg)]()
[![Plataforma](https://img.shields.io/badge/target-Raspberry%20Pi-green.svg)]()
[![Build](https://img.shields.io/badge/build-Yocto-kirkstone.svg)]()
[![Lenguajes](https://img.shields.io/badge/lenguajes-Python-orange.svg)]()
[![Stack](https://img.shields.io/badge/stack-OpenCV%20%7C%20TensorFlow%20Lite%20%7C%20V4L2%20%7C%20systemd-lightgrey.svg)]()

## Tabla de Contenido
- [1. Introducción](#1-introducción)
- [2. Justificación](#2-justificación)
- [3. Requerimientos](#3-requerimientos)
  - [3.1 Requerimientos funcionales (RF)](#31-requerimientos-funcionales-rf)
  - [3.2 Requerimientos no funcionales (RNF)](#32-requerimientos-no-funcionales-rnf)
  - [3.3 Interfaces y dependencias](#33-interfaces-y-dependencias)
  - [3.4 Criterios de aceptación](#34-criterios-de-aceptación)
- [4. Casos de uso](#4-casos-de-uso)
- [5. Arquitectura del sistema](#5-arquitectura-del-sistema)
- [6. Vista operacional y funcional](#6-vista-operacional-y-funcional)
- [7. Plan de trabajo y cronograma](#7-plan-de-trabajo-y-cronograma)
- [8. Integración y despliegue (Yocto)](#8-integración-y-despliegue-yocto)
- [9. Pruebas y validación](#9-pruebas-y-validación)
- [10. Métricas y observabilidad](#10-métricas-y-observabilidad)
- [11. Bitácora Christopher](#11-bitacora-christopher)
- [12. Bitácora Elena](#12-bitacora-elena)
- [13. Bitácora Kendy](#13-bitacora-kendy)
- [14. Documentación Entrega Preliminar](#14--documentación-entrega-preliminar)


---

## 📘 Descripción del Proyect 
El proyecto tiene como objetivo diseñar e implementar un sistema de **cruce inteligente** que utilice **Edge AI** (inteligencia artificial en el borde) para detectar vehículos, peatones o fauna, y optimizar el flujo del tráfico de forma autónoma.  
El sistema estará basado en **Raspberry Pi**, integrando **TensorFlow Lite** y **OpenCV**, dentro de una imagen Linux personalizada generada con **Yocto Project**.
=======

## 1. Introducción  
<p align="justify">
El aumento constante del parque vehicular y la expansión de las ciudades han generado una problemática creciente en la gestión del tránsito y la seguridad vial. En los cruces más concurridos, donde convergen peatones, ciclistas, motocicletas y vehículos particulares, los accidentes y la congestión se presentan con mayor frecuencia debido a la limitada capacidad de los sistemas tradicionales para adaptarse a las condiciones dinámicas del entorno urbano.
</p>
<p align="justify">
En este contexto, la inteligencia artificial embebida (Edge AI) surge como una alternativa tecnológica capaz de ejecutar algoritmos de visión por computador y aprendizaje automático directamente en dispositivos de bajo consumo energético, como el Raspberry Pi. Esta capacidad de procesamiento local permite realizar tareas de detección, clasificación y seguimiento de objetos en tiempo real sin depender completamente de la conectividad a la nube, lo que reduce la latencia y mejora la privacidad de los datos.
</p>
<p align="justify">
El presente proyecto propone el desarrollo de un sistema embebido que funcione como nodo inteligente dentro de una red de monitoreo de tránsito urbano. Cada nodo estará basado en hardware de Raspberry Pi con una cámara periférica y sensores complementarios, ejecutando modelos optimizados de visión artificial con TensorFlow Lite y OpenCV. El objetivo es detectar y clasificar peatones, ciclistas, fauna o vehículos, y con ello ofrecer información útil para la toma de decisiones en la gestión del tránsito y el mejoramiento de la seguridad vial.
</p>
<p align="justify">
El desarrollo de este tipo de sistemas no solo promueve la aplicación práctica de los conocimientos adquiridos en la asignatura de Sistemas Embebidos, sino que también alinea la formación del estudiantado con las tendencias actuales de la industria electrónica, donde convergen la inteligencia artificial, el IoT y la computación de borde.
</p>


## 2. Justificación
<p align="justify">
Las intersecciones urbanas concentran buena parte de las fricciones de la movilidad: trayectorias impredecibles de peatones y ciclistas, picos de congestión y decisiones de conducción tomadas bajo presión. En este entorno cambiante, los esquemas tradicionales de control —basados en temporizaciones fijas o conteos manuales— resultan insuficientes para anticipar comportamientos y reaccionar con la rapidez que exige la seguridad vial. Incorporar inteligencia en el borde (Edge AI) permite llevar el análisis al lugar donde ocurre el fenómeno, reduciendo la latencia, disminuyendo la dependencia de la nube y resguardando la privacidad de quienes transitan.
</p>

<p align="justify">
Este proyecto propone nodos embebidos basados en Raspberry Pi que ejecutan, en tiempo real, modelos livianos de visión por computador para detectar, clasificar y seguir peatones, ciclistas, fauna y vehículos. Al observar el flujo local con granularidad fina (escena a escena), el sistema puede proveer evidencia cuantitativa para ajustar fases semafóricas, activar alertas preventivas o caracterizar riesgos específicos del cruce (por ejemplo, puntos ciegos peatonales en determinadas horas). Así, la solución trasciende el conteo básico y se orienta a decisiones operativas informadas que impactan directamente en la reducción de incidentes y la mejora de la fluidez.
</p>

<p align="justify">
Desde la perspectiva tecnológica, la iniciativa articula un ecosistema embebido moderno: construcción de una imagen de Linux con Yocto Project, integración de OpenCV y TensorFlow Lite, y despliegue en hardware accesible. Esta combinación habilita ciclos de iteración cortos (medición–ajuste–validación) y una escalabilidad pragmática, pues es factible replicar nodos en múltiples cruces con costos razonables y mantenimiento estandarizado.
</p>

<p align="justify">
El proyecto también posee un alto valor formativo, al poner al equipo frente a retos reales de ingeniería: levantamiento y priorización de requerimientos, diseño de arquitecturas de hardware y software, manejo de dependencias, pruebas en campo y validación contra casos de uso. Así, el estudiantado fortalece competencias clave —diseño, integración, validación y documentación— alineadas con las demandas actuales de la industria electrónica y de sistemas embebidos.
</p>


En síntesis, la propuesta es pertinente por cuatro razones:

* Relevancia social: contribuye a la seguridad vial y a la movilidad sostenible en puntos críticos de la ciudad.

* Eficiencia operativa: habilita decisiones locales de baja latencia que mejoran el desempeño del cruce.

* Sostenibilidad tecnológica: se apoya en hardware de bajo costo y software modular, replicable y mantenible.

* Formación integral: consolida competencias profesionales con una experiencia aplicada de extremo a extremo.

Con base en lo anterior, en la siguiente sección se detallan los requerimientos funcionales y no funcionales que orientan el diseño y desarrollo del sistema propuesto.

## 3. Requerimientos

### 3.1 Requerimientos funcionales (RF)
- **RF1.** Capturar video en tiempo real desde cámara (CSI/USB, V4L2).
- **RF2.** Detectar y **clasificar** vehículos, peatones y fauna (TensorFlow Lite).
- **RF3.** **Seguimiento (tracking)** de objetos con IDs temporales.
- **RF4.** **Eventos** por cruce (conteos por clase, timestamps) y agregados (p.ej., por minuto).
- **RF5.** Exponer métricas localmente (CLI/log) y **publicar** a red (HTTP/MQTT).
- **RF6.** **Arranque autónomo**: service de la app al boot (systemd).
- **RF7.** **Registro de auditoría**: fallos, latencias, FPS y salud del nodo.

### 3.2 Requerimientos no funcionales (RNF)
- **RNF1.** Latencia “captura→detección→evento” ≤ **500 ms** (meta demo).
- **RNF2.** Throughput objetivo ≥ **10 FPS** sostenidos a **640×480**.
- **RNF3.** Robustez: recuperación si la cámara se desconecta/reconecta sin reiniciar.
- **RNF4.** **Observabilidad**: logs con niveles y métricas (CPU/RAM/FPS/colas).
- **RNF5.** **Despliegue reproducible** con **Yocto** (OpenCV, TFLite, drivers incluidos).
- **RNF6.** **Seguridad mínima**: no exponer servicios sin autenticación fuera de la LAN del demo.
- **RNF7.** **Mantenibilidad**: código/recetas en GitHub con README de build/instalación.

### 3.3 Interfaces y dependencias
- **I1.** Cámara CSI/USB vía **V4L2**.  
- **I2.** Red Ethernet/Wi-Fi para publicación de métricas/eventos.  
- **I3.** GPIO opcional (p.ej., LED indicador).  
- **D1.** **OpenCV** + **TensorFlow Lite**; **D2.** recetas Yocto; **D3.** servicio **systemd**.

### 3.4 Criterios de aceptación
- **CA1.** RF/RNF documentados y **rastreables** a objetivos.
- **CA2.** Casos de uso demostrados **end-to-end** en Raspberry Pi.
- **CA3.** Diagrama **HW/SW** con funciones→componentes e interfaces.
- **CA4.** **Build Yocto** reproducible con bitácora y árbol de dependencias.
- **CA5.** **Imagen bootea**, servicio corre, detección visible y métricas publicadas.
- **CA6.** **Plan** con hitos alineados a propuesta/demos.

---

## 4. Casos de uso

**Actores:** Operador (humano), Cámara (sensor), Servicio de Detección (app), Servicio de Publicación (red), Sistema de Registro (logs).

- **UC1. Inicializar nodo** — *Pre:* Imagen Yocto instalada. *Flujo:* Arranca SO → systemd lanza app → valida cámara/modelo. *Post:* Servicio “listo”.
- **UC2. Capturar video** — *Flujo:* Obtener frames ≥10 FPS; manejar errores de dispositivo. *Post:* Frames en buffer.
- **UC3. Detectar y clasificar objetos** — *Flujo:* Inferencia TFLite por frame → bboxes + clase. *Post:* Detecciones por frame.
- **UC4. Seguir objetos (tracking)** — *Flujo:* Asociar detecciones entre frames; asignar IDs; trayectorias. *Post:* Tracking activo.
- **UC5. Generar eventos y métricas** — *Flujo:* Conteos por clase/tiempo; KPIs. *Post:* Evento + KPIs actualizados.
- **UC6. Publicar datos a la red** — *Flujo:* HTTP/MQTT; reconexión si falla. *Post:* Confirmación o reintento.
- **UC7. Monitorear estado** — *Flujo:* Consultar FPS/latencia/CPU/RAM/estado cámara-red. *Post:* Salud verificada.
- **UC8. Gestionar fallos de cámara** — *Flujo:* Detectar desconexión; reabrir dispositivo; log. *Post:* Recuperación sin reboot.
- **UC9. Apagar/actualizar nodo** — *Flujo:* Detener servicio; actualizar imagen/paquete; reiniciar. *Post:* Nodo actualizado.
- **UC10. Demostración académica** — *Flujo:* Presentación end-to-end en Pi. *Post:* Evidencia para evaluación.

<p align="center">
  <img src="imagenes/Diagrama de caso de uso.png" alt="Casos de uso del sistema" width="600"/>
</p>

---

## 5. Arquitectura del sistema
> **TODO:** Diagrama HW/SW (Raspberry Pi, cámara, módulos ML, colas, red, logging, publicación).  
> **TODO:** Describir componentes, interfaces (V4L2, HTTP/MQTT, systemd) y flujos de datos.
---

## 6. Vista operacional y funcional
> **TODO:** Escenarios de operación, estados del nodo, flujos (captura→detección→tracking→eventos→publicación), manejo de errores.
---

## 7. Plan de trabajo y cronograma
> **TODO:** Incluir Gantt y checklist de actividades/hitos hasta la demo.  
> **Hitos sugeridos:** Propuesta → Arquitectura → Recetas Yocto → Imagen → Pruebas en Pi → Demo.
---

## 8. Integración y despliegue (Yocto)

### 8.1 Requisitos de build
> **TODO:** Host, toolchain, caches (DL_DIR/SSTATE_DIR), ramas y versiones.
### 8.2 Capas y recetas
> **TODO:** meta, meta-poky, meta-yocto-bsp, meta-openembedded/meta-oe, capa propia (app + deps TFLite/OpenCV).
### 8.3 Servicio de la aplicación (systemd)
> **TODO:** `*.service` con `Restart=always`, env vars, logs a journald/archivo.
### 8.4 Imagen y artefactos
> **TODO:** Tipo de imagen, tamaño/particiones, método de flasheo, validación post-flash.
---

## 9. Pruebas y validación
- **Unitarias:** parsers, colas, post-proc.  
- **Integración:** cámara→detección→eventos→publicación.  
- **Desempeño:** FPS, latencia.  
- **Robustez:** desconexión/reconexión de cámara, red intermitente.  
> **TODO:** Plan de pruebas con IDs (T-XXX), datos de prueba, criterios de pase/fallo y scripts.
---

## 10. Métricas y observabilidad
- **KPIs mínimos:** FPS, latencia, CPU/RAM, tamaño de cola, tasa de publicación, conteos por clase.  
> **TODO:** Formato de logs (JSON/CSV), niveles, ejemplo de export (HTTP/MQTT), dashboard local si aplica.


---
## 14. 📚 Documentación Entrega Preliminar
- [Cronograma del proyecto](docs/CRONOGRAMA.md)
- [Información General del Sistema](docs/INFORMACION-GENERAL.md)
- [Información Técnica del Sistema](docs/INFORMACION-TECNICA.md)
- [Bitácora de Kendy](docs/BITACORA-KENDY.md)
- [Bitácora de Elena](docs/BITACORA-ELENA.md)
- [Bitácora de Chris](docs/BITACORA-CHRIS.md)


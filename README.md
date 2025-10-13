# 🚦 Proyecto 2 – Cruces Inteligentes con Edge AI  

> Nodo de visión en el borde para detección, clasificación y conteo en cruces viales con Raspberry Pi + Yocto + OpenCV/TensorFlow Lite.

[![Estado](https://img.shields.io/badge/estado-en_desarrollo-blue.svg)]()
[![Plataforma](https://img.shields.io/badge/target-Raspberry%20Pi-green.svg)]()
[![Build](https://img.shields.io/badge/build-Yocto-kirkstone.svg)]()
[![Lenguajes](https://img.shields.io/badge/lenguajes-Python-orange.svg)]()
[![Stack](https://img.shields.io/badge/stack-OpenCV%20%7C%20TensorFlow%20Lite%20%7C%20V4L2%20%7C%20systemd-lightgrey.svg)]()

## 📘 Descripción del Proyecto  
El proyecto tiene como objetivo diseñar e implementar un sistema de **cruce inteligente** que utilice **Edge AI** (inteligencia artificial en el borde) para detectar vehículos, peatones o fauna, y optimizar el flujo del tráfico de forma autónoma.  
El sistema estará basado en **Raspberry Pi**, integrando **TensorFlow Lite** y **OpenCV**, dentro de una imagen Linux personalizada generada con **Yocto Project**.

---

## 📚 Documentación Entrega Preliminar
- [Cronograma del proyecto](docs/CRONOGRAMA.md)
- [Información General del Sistema](docs/INFORMACION-GENERAL.md)
- [Información Técnica del Sistema](docs/INFORMACION-TECNICA.md)
- [Bitácora de Kendy](docs/BITACORA-KENDY.md)
- [Bitácora de Elena](docs/BITACORA-ELENA.md)
- [Bitácora de Chris](docs/BITACORA-CHRIS.md)

---

## 📑 Tabla de Contenido
- [1. Introducción](#1-introducción)
- [2. Síntesis del problema](#2-síntesis-del-problema)
- [3. Alcance](#3-alcance)
- [4. Requerimientos](#4-requerimientos)
  - [4.1 Requerimientos funcionales (RF)](#41-requerimientos-funcionales-rf)
  - [4.2 Requerimientos no funcionales (RNF)](#42-requerimientos-no-funcionales-rnf)
  - [4.3 Interfaces y dependencias](#43-interfaces-y-dependencias)
  - [4.4 Criterios de aceptación](#44-criterios-de-aceptación)
  - [4.5 Matriz de rastreabilidad](#45-matriz-de-rastreabilidad)
- [5. Casos de uso](#5-casos-de-uso)
- [6. Arquitectura del sistema](#6-arquitectura-del-sistema)
- [7. Vista operacional y funcional](#7-vista-operacional-y-funcional)
- [8. Plan de trabajo y cronograma](#8-plan-de-trabajo-y-cronograma)
- [9. Integración y despliegue (Yocto)](#9-integración-y-despliegue-yocto)
- [10. Pruebas y validación](#10-pruebas-y-validación)
- [11. Métricas y observabilidad](#11-métricas-y-observabilidad)
- [12. Gestión de riesgos](#12-gestión-de-riesgos)
- [13. Entregables y demo](#13-entregables-y-demo)
- [14. Bitácoras y repositorio](#14-bitácoras-y-repositorio)
- [15. Anexos](#15-anexos)
- [16. Referencias](#16-referencias)
- [Licencia](#licencia)

---

## 1. Introducción
> **TODO:** Contexto, motivación, impacto, relación con sistemas embebidos y visión general.

## 2. Síntesis del problema
> **TODO:** 3–6 líneas que sinteticen el problema principal a resolver en el cruce inteligente.

## 3. Alcance
> **TODO:** Definir claramente qué **sí** incluye el MVP y qué **no** (fuera de alcance).

---

## 4. Requerimientos

### 4.1 Requerimientos funcionales (RF)
- **RF1.** Capturar video en tiempo real desde cámara (CSI/USB, V4L2).
- **RF2.** Detectar y **clasificar** vehículos, peatones y fauna (TensorFlow Lite).
- **RF3.** **Seguimiento (tracking)** de objetos con IDs temporales.
- **RF4.** **Eventos** por cruce (conteos por clase, timestamps) y agregados (p.ej., por minuto).
- **RF5.** Exponer métricas localmente (CLI/log) y **publicar** a red (HTTP/MQTT).
- **RF6.** **Arranque autónomo**: service de la app al boot (systemd).
- **RF7.** **Registro de auditoría**: fallos, latencias, FPS y salud del nodo.

### 4.2 Requerimientos no funcionales (RNF)
- **RNF1.** Latencia “captura→detección→evento” ≤ **500 ms** (meta demo).
- **RNF2.** Throughput objetivo ≥ **10 FPS** sostenidos a **640×480**.
- **RNF3.** Robustez: recuperación si la cámara se desconecta/reconecta sin reiniciar.
- **RNF4.** **Observabilidad**: logs con niveles y métricas (CPU/RAM/FPS/colas).
- **RNF5.** **Despliegue reproducible** con **Yocto** (OpenCV, TFLite, drivers incluidos).
- **RNF6.** **Seguridad mínima**: no exponer servicios sin autenticación fuera de la LAN del demo.
- **RNF7.** **Mantenibilidad**: código/recetas en GitHub con README de build/instalación.

### 4.3 Interfaces y dependencias
- **I1.** Cámara CSI/USB vía **V4L2**.  
- **I2.** Red Ethernet/Wi-Fi para publicación de métricas/eventos.  
- **I3.** GPIO opcional (p.ej., LED indicador).  
- **D1.** **OpenCV** + **TensorFlow Lite**; **D2.** recetas Yocto; **D3.** servicio **systemd**.

### 4.4 Criterios de aceptación
- **CA1.** RF/RNF documentados y **rastreables** a objetivos.
- **CA2.** Casos de uso demostrados **end-to-end** en Raspberry Pi.
- **CA3.** Diagrama **HW/SW** con funciones→componentes e interfaces.
- **CA4.** **Build Yocto** reproducible con bitácora y árbol de dependencias.
- **CA5.** **Imagen bootea**, servicio corre, detección visible y métricas publicadas.
- **CA6.** **Plan** con hitos alineados a propuesta/demos.

---

## 5. Casos de uso

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

## 6. Arquitectura del sistema
> **TODO:** Diagrama HW/SW (Raspberry Pi, cámara, módulos ML, colas, red, logging, publicación).  
> **TODO:** Describir componentes, interfaces (V4L2, HTTP/MQTT, systemd) y flujos de datos.

---

## 7. Vista operacional y funcional
> **TODO:** Escenarios de operación, estados del nodo, flujos (captura→detección→tracking→eventos→publicación), manejo de errores.

---

## 8. Plan de trabajo y cronograma
> **TODO:** Incluir Gantt y checklist de actividades/hitos hasta la demo.  
> **Hitos sugeridos:** Propuesta → Arquitectura → Recetas Yocto → Imagen → Pruebas en Pi → Demo.

---

## 9. Integración y despliegue (Yocto)

### 9.1 Requisitos de build
> **TODO:** Host, toolchain, caches (DL_DIR/SSTATE_DIR), ramas y versiones.

### 9.2 Capas y recetas
> **TODO:** meta, meta-poky, meta-yocto-bsp, meta-openembedded/meta-oe, capa propia (app + deps TFLite/OpenCV).

### 9.3 Servicio de la aplicación (systemd)
> **TODO:** `*.service` con `Restart=always`, env vars, logs a journald/archivo.

### 9.4 Imagen y artefactos
> **TODO:** Tipo de imagen, tamaño/particiones, método de flasheo, validación post-flash.

---

## 10. Pruebas y validación
- **Unitarias:** parsers, colas, post-proc.  
- **Integración:** cámara→detección→eventos→publicación.  
- **Desempeño:** FPS, latencia.  
- **Robustez:** desconexión/reconexión de cámara, red intermitente.  
> **TODO:** Plan de pruebas con IDs (T-XXX), datos de prueba, criterios de pase/fallo y scripts.

---

## 11. Métricas y observabilidad
- **KPIs mínimos:** FPS, latencia, CPU/RAM, tamaño de cola, tasa de publicación, conteos por clase.  
> **TODO:** Formato de logs (JSON/CSV), niveles, ejemplo de export (HTTP/MQTT), dashboard local si aplica.

---

## 12. Gestión de riesgos
> **TODO:** Tabla de riesgos (probabilidad, impacto, mitigación): rendimiento, iluminación, oclusiones, pérdida de cámara, fallos de red, tamaño de imagen, tiempos de build.

---

## 13. Entregables y demo
- **Entregables:** Propuesta, diagramas, código y recetas, bitácoras, imagen Yocto, artefactos, script de demo.  
- **Demo final (en Pi):** Captura, detección/clasificación, tracking, publicación y monitoreo en tiempo real.  
> **TODO:** Guion de demo (paso a paso) y criterios de evaluación.

---

## 14. Bitácoras y repositorio
**Estructura sugerida:**

# 📝 Bitácora de Kendy Arias  
### Proyecto: **Cruces Inteligentes con Edge AI**

Esta bitácora documenta el proceso completo de desarrollo del nodo inteligente basado en Raspberry Pi + YOLO + Yocto, desde la propuesta inicial hasta la integración final del sistema.

---

## 📅 03/10/2025
- Creación del repositorio principal `TSE-CrucesInteligentes`.
- Estructura base del proyecto (`docs/`, `src/`, `README.md`).

---

## 📅 09/10/2025
- Agregados los miembros del equipo al repositorio.
- Preparación de documentación colaborativa.

---

## 📅 10/10/2025
- Primer avance de la entrega preliminar (puntos 6, 7, 8 y 9).
- Se agregaron:
  - Bitácoras individuales  
  - Cronograma  
  - Información general  
  - Información técnica inicial  

---

## 📅 14/10/2025
- Finalizada mi parte de la **Información Técnica** de la propuesta inicial.

---

## 📅 15/10/2025
- Creación de una **imagen mínima Yocto**, pero las cámaras no funcionaban → faltaban dependencias.

---

## 📅 19/10/2025
- Segunda imagen mínima Yocto funcional en Raspberry Pi.
- Investigación de módulos de detección y del sistema de semáforo.

---

## 📅 20/10/2025
- Finalización de la sección de cámaras y hardware para la propuesta.
- Envío del documento completo de la propuesta con aportes de todo el equipo.

---

## 📅 10/11/2025
- Desarrollo del **script base de control del semáforo**, considerando entradas futuras de los detectores.

---

## 📅 16/11/2025
- Prueba de una imagen enviada por un compañero.
- Identificación de errores: faltaban recetas y servicios Yocto.
- Se determinó que era necesario integrar correctamente los scripts.

---

## 📅 17/11/2025
- Revisión y copia local de los scripts.
- Identificación de múltiples fallos de integración.
- Adaptación del script del semáforo.
- Creación de un **script maestro** para coordinar detectores y controlador.
- Éxito inicial: detecciones funcionando y flags generándose correctamente.

---

## 📅 22/11/2025
- Desarrollo de una **Interfaz Gráfica (GUI)** opcional para monitoreo del sistema.

---

## 📅 23/11/2025
- Creación de la primera **imagen Yocto completa** con:
  - servicio,
  - receta,
  - script de arranque,
  - aplicación integrada.
- Al probarla en la Raspberry Pi:
  - El servicio y el semáforo funcionaban → ✔  
  - Los detectores fallaban → ✘ (modelos diseñados para GPU)
- Adaptación de los detectores para CPU.
- Aún sin funcionar en la Pi por problemas adicionales.

---

## 📅 25/11/2025
- Descubrimiento del fallo principal: **la imagen estaba mal flasheada**.
- Tras flasheo correcto, el sistema arrancó sin errores, pero la detección seguía fallando.
- Se identificó la causa real:
  - **incompatibilidad entre YOLO y OpenCV 4.5.5 (Yocto)**.

---

## 📅 26/11/2025
- Encontrado un modelo compatible: **YOLOv5n ONNX para CPU**.
- Prueba exitosa en Raspberry Pi usando una imagen mínima + app.
- Ajustes SSH:
  - corrección de bounding boxes,
  - escalado,
  - flags,
  - envío al controlador.
- Los módulos funcionan al 100 %.
- La GUI queda lista, pero no se integra por:
  - tiempo limitado,
  - estabilidad de la imagen final.

---

## 📅 30/11/2025
- Actualización del documento final del proyecto.
- Actualización de bitácora.
- Subida al repositorio de:
  - scripts finales,
  - archivos Yocto,
  - documentación.
- Grabación del video final con demostración del sistema.
- Como extra: demostración de la GUI en funcionamiento.

---

## ✅ Estado Final del Proyecto
El sistema está completamente funcional en Raspberry Pi, con:
- Detección de peatones, vehículos y animales.
- Tres procesos independientes que se comunican mediante `/tmp`.
- Control autónomo del semáforo mediante máquina de estados.
- Modelo de detección 100 % compatible con OpenCV 4.5.5.
- Arquitectura modular lista para futuras extensiones (GUI, IoT, MQTT, etc.).

---


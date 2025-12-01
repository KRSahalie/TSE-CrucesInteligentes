# Bitácora de Chris
---

## 7/10/2025
- Se determinaron los requisitos que debe cumplir el sistema.  
- Desarrollo del diagrama de casos de uso.  
- Creación y establecimiento de los task en el diagrama de Gantt.  

## 9/10/2025
- Configuración del entorno de desarrollo en Yocto.  
- Verificación de dependencias y herramientas del sistema embebido.  
- Actualización del repositorio en GitHub.

## 13/10/2025
- Actualización del repositorio en GitHub.

## 19/10/2025
- Desarrollo de imagen minima para Raspberry pi 4
- Desarrollo del script en pythton para deteccion de vehiculos y peatones.
- Actualizacion de la documentacion en el repositorio.

## 25/10/2025
- Correccion de erroes.

## 9/11/2025
- Reformulacion del programa en python para funcionamiento 100% con 2 camaras.

## 12/11/2025
- Separación del código en dos módulos independientes: `people_counter_cam.py` (peatones) y `veh_counter_cam.py` (vehículos).  
- Ajuste de los parámetros de tamaño de imagen, umbral de confianza y eliminación de cajas superpuestas.  
- Primera validación del conteo de personas.

## 14/11/2025
- Implementación de la lógica de máscaras para limitar las detecciones.  
- Pruebas de detección de peatones con diferentes resoluciones de cámara.  

## 17/11/2025
- Implementacion de la logica para distinguir vehículos en movimiento de vehículos detenidos.  
- Primera validación del conteo de vehiculos en movimiento.

## 19/11/2025
- Implementación y ajuste de las banderas de salida (archivos de flag).  
- Calibración de umbrales de decisión para el encendido/apagado de las banderas.  

## 22/11/2025
- Pruebas integrales de `people_counter_cam.py` y `veh_counter_cam.py` ejecutándose en la Raspberry Pi 4 con la imagen de Yocto.  
- Ajustes finales de parámetros de confianza, tamaño de imagen y frecuencias de detección para asegurar funcionamiento estable en tiempo real.  
- Documentación detallada del flujo de trabajo de ambos scripts.

## Estado Actual
- Proyecto presentado.

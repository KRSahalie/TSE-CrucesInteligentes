# La información general abarca los puntos 1,2,3,4 y 5 de la entrega preliminar.

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

**Actores:** Operador (humano), Vehiculos, administrador.

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
  <img src="https://github.com/KRSahalie/TSE-CrucesInteligentes/blob/main/imagenes/Diagrama%20de%20caso%20de%20uso.png" alt="Diagrama de casos de uso" width="600">
</p>


Agregando...

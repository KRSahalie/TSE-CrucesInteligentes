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

## Casos de uso por actor
### Administrador

1. **Configurar parámetros del sistema**  
   Ajusta umbrales de detección (distancia, confianza de modelo, umbrales de cola), temporización de fases y permisos de red.  
   _Post:_ Sistema actualizado y registrado en logs.

2. **Reiniciar o actualizar software**  
   Ejecuta reinicio seguro del nodo, actualiza imagen Yocto o paquete de la app Edge AI.  
   _Post:_ Nodo operativo con última versión.

4. **Consultar reportes de tráfico**  
   Accede a históricos de conteos vehiculares y cruces peatonales.  
   _Post:_ Informes listos para análisis y toma de decisiones.


### Peatón

1. **Solicitar paso peatonal**  
   Presiona el botón o sensor capacitivo → el sistema agenda la fase segura de cruce.  
   _Post:_ Semáforo peatonal verde y temporizador visible.

2. **Alertar situación de riesgo**  
   Botón de emergencia u orden manual si se detecta un vehículo invasor.  
   _Post:_ Evento registrado y alerta al administrador.

3. **Consultar tiempo restante de cruce**  
   Pantalla o voz sintética indica cuánto falta para cerrar la fase.  
   _Post:_ Información accesible para personas con discapacidad.

### Vehículo (conductor)

1. **Detectar aproximación**  
   El sistema de visión detecta el vehículo y clasifica tipo (auto, bus, moto, emergencia).  
   _Post:_ Demanda vehicular actualizada.

2. **Esperar cambio de color (fase)**  
   Si el semáforo está en rojo, mantiene la columna de espera según la cola detectada.  
   _Post:_ Tiempo de espera monitoreado y reportado.

3. **Vehículo de emergencia prioritario**  
   Detección por sirena/luces → se otorga fase preferente.  
   _Post:_ Tránsito priorizado y evento registrado.

4. **Infracción por cruce en rojo**  
   Vehículo entra en rojo → el sistema genera evidencia (captura + timestamp).  
   _Post:_ Reporte enviado al administrador y almacenado en logs.


<p align="center">
  <img src="https://github.com/KRSahalie/TSE-CrucesInteligentes/blob/main/imagenes/Diagrama%20de%20caso%20de%20uso.png" alt="Diagrama de casos de uso" width="600">
</p>


Agregando...

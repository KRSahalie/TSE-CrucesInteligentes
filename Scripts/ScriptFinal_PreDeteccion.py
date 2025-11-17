import time       # Para manejar pausas (time.sleep)
import datetime   # Para generar la marca de tiempo (timestamp)
import os        # Para trabajar con archivos de bandera (flags)
import json       # Incluido, aunque no usado directamente, es estándar en I/O

# ----------------------------------------------------------------------
# --- CONFIGURACIÓN GLOBAL DEL SISTEMA Y PARÁMETROS ---
# ----------------------------------------------------------------------
LOG_FILE = "traffic_control_log.csv" 
# Nombre del archivo de registro de eventos. Formato: CSV (Comma Separated Values).

# --- ARCHIVOS DE BANDERA (FLAGS) PRODUCIDOS POR LOS DETECTORES ---
PEDESTRIAN_FLAG_FILE = "/tmp/ped_flag.txt"      # Escrito por people_counter_cam.py (0/1)
ANIMAL_FLAG_FILE     = "/tmp/animal_flag.txt"  # Escrito por veh_counter_cam.py (0/1)


# --- PARÁMETROS DE TIEMPO (SEGUNDOS) ---
TIEMPO_VERDE_CARROS = 10     # Duración estándar de la luz VERDE para vehículos
TIEMPO_AMARILLO_CARROS = 2   # Duración de la luz AMARILLA
TIEMPO_ROJO_NORMAL = 20      # Duración del ROJO (Simula el tráfico opuesto)
TIEMPO_ROJO_PEA = 5          # Duración del VERDE para peatones
TIEMPO_ROJO_ANIMAL = 5       # Duración del bloqueo total por animal

# ----------------------------------------------------------------------
# --- CLASES DE ESTADO DEL SEMÁFORO ---
# ----------------------------------------------------------------------

class SemaforoCarros:
    """Representa el estado del semáforo vehicular."""
    def __init__(self):
        self.estado = "ROJO" # Estado inicial
    def set_estado(self, nuevo_estado):
        self.estado = nuevo_estado

class SemaforoPeatones:
    """Representa el estado del semáforo peatonal."""
    def __init__(self):
        self.estado = "ROJO" # El peatonal siempre debe iniciar en ROJO
    def set_estado(self, nuevo_estado):
        self.estado = nuevo_estado

# ----------------------------------------------------------------------
# --- MÓDULO DE SALIDA Y REGISTRO ESTÁNDAR (LOGGING) ---
# ----------------------------------------------------------------------

def guardar_log(level, module, message, estado_carros, estado_peatones):
    """
    Registra el evento en formato Syslog/CSV para recolección de datos.
    
    Args:
        level (str): Nivel de criticidad (INFO, WARNING, CRITICAL).
        module (str): Módulo o tipo de evento (CAR_STATE, PEA_PASS, ANIMAL_ALERT).
        message (str): Descripción concisa del evento.
        estado_carros (str): Estado actual del semáforo vehicular.
        estado_peatones (str): Estado actual del semáforo peatonal.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Construcción de la línea de log en formato CSV
    log_entry = f"{timestamp},{level},{module},{estado_carros},{estado_peatones},\"{message}\""
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_entry + "\n")
    except IOError as e:
        # Manejo de error crítico de escritura de archivo
        print(f"CRITICAL ERROR: No se pudo escribir en log. {e}")

    # Salida MÍNIMA a la terminal (para monitoreo en vivo)
    print(f"[{timestamp}] [{level:<7}] {module:<11}: {message}")


# ----------------------------------------------------------------------
# --- CONTROLADOR PRINCIPAL DEL SEMÁFORO ---
# ----------------------------------------------------------------------

class ControladorSemaforos:
    def __init__(self):
        self.carros = SemaforoCarros()
        self.peatones = SemaforoPeatones()
        guardar_log("INFO", "SYSTEM_INIT", "Traffic control started", self.carros.estado, self.peatones.estado)
        # Bandera para rastrear en qué estado se interrumpió el ciclo.
        # Es importante para la función 'ciclo_normal' al reanudar.
        self.interrupted_at = None 

    def paso_peatones(self):
        """Gestiona la interrupción del ciclo por detección de peatones."""
        
        # Regla 1: Transición segura si los carros están en VERDE
        if self.carros.estado == "VERDE":
            self.interrupted_at = "VERDE"
            self.carros.set_estado("AMARILLO")
            guardar_log("WARNING", "PEA_ALERT", "Car transition to AMBER", self.carros.estado, self.peatones.estado)
            time.sleep(TIEMPO_AMARILLO_CARROS)
        elif self.carros.estado == "ROJO":
            # Si se interrumpe durante el ROJO, forzamos la reanudación a VERDE
            self.interrupted_at = "ROJO"
        
        # Regla 2: Bloqueo Vehicular (VERDE Peatonal)
        self.carros.set_estado("ROJO")
        self.peatones.set_estado("VERDE")
        guardar_log("INFO", "PEA_PASS", f"Active duration {TIEMPO_ROJO_PEA}s", self.carros.estado, self.peatones.estado)
        
        for _ in range(TIEMPO_ROJO_PEA):
            time.sleep(1)

        # Regla 3: Finalización segura
        self.peatones.set_estado("ROJO")
        guardar_log("INFO", "PEA_END", "Resuming traffic cycle", self.carros.estado, self.peatones.estado)


    def interrupcion_animal(self):
        """Gestiona la interrupción crítica por detección de animales."""
        
        # *** MEJORA: Optimización de ANIMAL_BLOCK en estado ROJO ***
        if self.carros.estado == "ROJO":
            # Si el tráfico ya está detenido (ROJO), no se realiza la secuencia AMBAR/ROJO
            # Solo se registra y se asegura que peatones estén ROJO
            if self.peatones.estado == "VERDE":
                self.peatones.set_estado("ROJO")
                guardar_log("CRITICAL", "ANIMAL_SAFETY", "Pedestrian forced to RED due to animal threat", self.carros.estado, self.peatones.estado)
            
            self.interrupted_at = "ROJO"
            guardar_log("WARNING", "ANIMAL_DETECT", "Animal detected, cars already RED. Holding position.", self.carros.estado, self.peatones.estado)
            
            for _ in range(TIEMPO_ROJO_ANIMAL):
                time.sleep(1)

        else: # Si el carro está en VERDE o AMARILLO (Requiere transición segura)
            # 1. Transición de Advertencia
            self.interrupted_at = "VERDE"
            self.carros.set_estado("AMARILLO")
            guardar_log("CRITICAL", "ANIMAL_ALERT", "Critical transition to AMBER", self.carros.estado, self.peatones.estado)
            time.sleep(TIEMPO_AMARILLO_CARROS)

            # 2. Bloqueo Crítico Total
            self.carros.set_estado("ROJO")
            self.peatones.set_estado("ROJO") 
            guardar_log("CRITICAL", "ANIMAL_BLOCK", f"Road blocked duration {TIEMPO_ROJO_ANIMAL}s", self.carros.estado, self.peatones.estado)
            
            for _ in range(TIEMPO_ROJO_ANIMAL):
                time.sleep(1)

        # 3. Finalización y Reanudación
        guardar_log("INFO", "ANIMAL_END", "Blockage resolved. Resuming traffic cycle", self.carros.estado, self.peatones.estado)
        
    def pausa_con_monitoreo(self, duracion, obtener_senal):
        """
        Detiene la ejecución por 'duracion' segundos, revisando la detección 
        de señales en cada segundo (monitoreo activo).
        """
        for _ in range(duracion):
            senal = obtener_senal()
            if senal == "PEATON":
                self.paso_peatones()
                return # Sale de la pausa y obliga al ciclo a reanudar
            elif senal == "ANIMAL":
                self.interrupcion_animal()
                return # Sale de la pausa y obliga al ciclo a reanudar
            time.sleep(1)

    def ciclo_normal(self, obtener_senal):
        """Define la secuencia estándar de un semáforo (ROJO -> VERDE -> AMARILLO)."""
        
        # *** MEJORA: Manejo de Reanudación de Ciclo ***
        if self.interrupted_at == "ROJO":
            # Si una interrupción terminó cuando el sistema estaba en la fase ROJO
            # (ej. en la pausa de 20s), saltamos el primer paso ROJO para pasar a VERDE.
            self.interrupted_at = None
        else:
            # 1. Estado ROJO (Paso normal)
            self.carros.set_estado("ROJO")
            guardar_log("INFO", "CAR_STATE", f"Switch to RED. Wait {TIEMPO_ROJO_NORMAL}s", self.carros.estado, self.peatones.estado)
            self.pausa_con_monitoreo(TIEMPO_ROJO_NORMAL, obtener_senal)
        
        # 2. Estado VERDE
        # La condición verifica que el estado actual sea ROJO, asegurando que si 
        # hubo una interrupción en AMARILLO/VERDE, el sistema ya pasó por ROJO 
        # durante la interrupción y puede avanzar.
        if self.carros.estado == "ROJO": 
            self.carros.set_estado("VERDE")
            guardar_log("INFO", "CAR_STATE", f"Switch to GREEN. Active {TIEMPO_VERDE_CARROS}s", self.carros.estado, self.peatones.estado)
            self.pausa_con_monitoreo(TIEMPO_VERDE_CARROS, obtener_senal)
        
        # 3. Estado AMARILLO
        if self.carros.estado == "VERDE": 
            self.carros.set_estado("AMARILLO")
            guardar_log("INFO", "CAR_STATE", f"Switch to AMBER. Transition {TIEMPO_AMARILLO_CARROS}s", self.carros.estado, self.peatones.estado)
            self.pausa_con_monitoreo(TIEMPO_AMARILLO_CARROS, obtener_senal)
        
        # Reinicia la bandera de interrupción al final de la secuencia de ciclo
        self.interrupted_at = None 
        
# ----------------------------------------------------------------------
# --- LECTURA DE FLAGS DE LOS DETECTORES (MODO REAL) ---
# ----------------------------------------------------------------------

def leer_flag(path, default=0):
    """Lee un archivo de bandera ("0"/"1"). Si no existe o hay error, devuelve default."""
    try:
        with open(path, "r") as f:
            raw = f.read().strip()
        return 1 if raw == "1" else 0
    except FileNotFoundError:
        # Si el archivo aún no existe, asumimos que la condición no está activa.
        return default
    except Exception:
        # En producción podríamos registrar el error con guardar_log; aquí lo suavizamos.
        return default

def obtener_senal():
    """
    Consulta las banderas generadas por los detectores:
    - ANIMAL_FLAG_FILE tiene prioridad sobre PEDESTRIAN_FLAG_FILE.
    Retorna:
        "ANIMAL" si hay animal en vía,
        "PEATON" si hay demanda peatonal,
        None si no hay eventos.
    """
    animal_flag = leer_flag(ANIMAL_FLAG_FILE, default=0)
    ped_flag    = leer_flag(PEDESTRIAN_FLAG_FILE, default=0)

    # Prioridad de seguridad: primero ANIMAL
    if animal_flag == 1:
        return "ANIMAL"
    if ped_flag == 1:
        return "PEATON"
    return None


# ----------------------------------------------------------------------
# --- EJECUCIÓN PRINCIPAL DEL SCRIPT ---
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Inicializa el controlador al arrancar el script
    controlador = ControladorSemaforos()

    try:
        # Bucle infinito para mantener el sistema de control en funcionamiento
        while True:
            controlador.ciclo_normal(obtener_senal)
    except KeyboardInterrupt:
        # Captura la señal de detención manual (Ctrl+C)
        guardar_log("WARNING", "SYSTEM_SHUTDOWN", "System stopped by user (KeyboardInterrupt)", "N/A", "N/A")
        print("\n--- OPERACIÓN DETENIDA ---")
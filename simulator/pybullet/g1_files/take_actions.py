# take_actions.py

from pynput.keyboard import Controller
keyboard = Controller()

# --- Funciones para el Experimento de Swaying ---

def start_swaying(sim_state):
    """Inicia el movimiento de swaying pulsando la tecla '1'."""
    print(f"\n[{sim_state['current_time']:.2f}s] INICIANDO SWAYING...\n")
    return 
    keyboard.press('1')
    keyboard.release('1')

def start_walking(sim_state):
    print(f"\n[{sim_state['current_time']:.2f}s] INICIANDO FORWARD WALKING...\n")
    keyboard.press('8')
    keyboard.release('8')

def set_frontal_force(sim_state):
    """Aplica una fuerza frontal de -25N en X."""
    print(f"\n[{sim_state['current_time']:.2f}s] Aplicando FUERZA FRONTAL...\n")
    sim_state['current_force_target'] = [-15.0, 0.0, 0.0]
    sim_state['t_force_start'] = sim_state['current_time']

def set_lateral_force(sim_state):
    """Aplica una fuerza lateral de 25N en Y."""
    print(f"\n[{sim_state['current_time']:.2f}s] Aplicando FUERZA LATERAL...\n")
    sim_state['current_force_target'] = [0.0, 10.0, 0.0]
    sim_state['t_force_start'] = sim_state['current_time']

def set_combined_force(sim_state):
    """NUEVA: Aplica una fuerza bidireccional."""
    print(f"\n[{sim_state['current_time']:.2f}s] Aplicando FUERZA BIDIRECCIONAL...\n")
    sim_state['current_force_target'] = [25.0, -10.0, 0.0]
    sim_state['t_force_start'] = sim_state['current_time']

def reset_forces(sim_state):
    """Quita todas las fuerzas externas."""
    print(f"\n[{sim_state['current_time']:.2f}s] Reseteando fuerzas (Pausa de Recuperación)...\n")
    sim_state['current_force_target'] = [0.0, 0.0, 0.0]
    sim_state['t_force_start'] = sim_state['current_time']
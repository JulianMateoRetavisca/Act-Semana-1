import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import random

# Configuración del laberinto
FILAS = 12  # Laberinto más grande
COLUMNAS = 12
INICIO = (0, 0)
META = (11, 11)

# Tabla Q global
Q_table = {}
OBSTACULOS = []  # Se generará aleatoriamente

def generar_laberinto_aleatorio(filas=12, columnas=12, densidad=0.30):
    """
    Genera un laberinto aleatorio usando un algoritmo mejorado.
    densidad: porcentaje de celdas que serán obstáculos (0.0 - 0.5)
    """
    global OBSTACULOS
    obstaculos = set()
    
    # Asegurar que inicio y meta estén libres
    celdas_prohibidas = {INICIO, META}
    
    # Generar obstáculos aleatorios
    total_celdas = filas * columnas
    num_obstaculos = int(total_celdas * densidad)
    
    # Crear paredes estructuradas (patrón de laberinto)
    # Paredes horizontales aleatorias
    for i in range(1, filas - 1, 2):
        inicio_pared = random.randint(0, columnas - 5)
        longitud = random.randint(3, 6)
        for j in range(inicio_pared, min(inicio_pared + longitud, columnas)):
            if (i, j) not in celdas_prohibidas:
                obstaculos.add((i, j))
    
    # Paredes verticales aleatorias
    for j in range(1, columnas - 1, 2):
        inicio_pared = random.randint(0, filas - 5)
        longitud = random.randint(3, 6)
        for i in range(inicio_pared, min(inicio_pared + longitud, filas)):
            if (i, j) not in celdas_prohibidas:
                obstaculos.add((i, j))
    
    # Agregar obstáculos aleatorios adicionales para mayor dificultad
    while len(obstaculos) < num_obstaculos:
        i = random.randint(0, filas - 1)
        j = random.randint(0, columnas - 1)
        if (i, j) not in celdas_prohibidas and (i, j) not in obstaculos:
            # Verificar que no bloqueamos completamente el camino
            if not bloquea_camino(obstaculos | {(i, j)}, filas, columnas):
                obstaculos.add((i, j))
    
    OBSTACULOS = list(obstaculos)
    return OBSTACULOS

def bloquea_camino(obstaculos, filas, columnas):
    """Verifica si el laberinto tiene solución usando BFS"""
    from collections import deque
    
    visitados = set()
    cola = deque([INICIO])
    visitados.add(INICIO)
    
    while cola:
        estado = cola.popleft()
        
        if estado == META:
            return False  # Hay camino, no está bloqueado
        
        # Explorar vecinos
        i, j = estado
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            nuevo_estado = (ni, nj)
            
            if (0 <= ni < filas and 0 <= nj < columnas and 
                nuevo_estado not in obstaculos and 
                nuevo_estado not in visitados):
                visitados.add(nuevo_estado)
                cola.append(nuevo_estado)
    
    return True  # No hay camino, está bloqueado

def inicializar_q_table():
    """Inicializa la tabla Q con ceros"""
    global Q_table
    Q_table = {}
    for i in range(FILAS):
        for j in range(COLUMNAS):
            if (i, j) not in OBSTACULOS:
                Q_table[(i, j)] = {'arriba': 0, 'abajo': 0, 'izquierda': 0, 'derecha': 0}
    return Q_table

def obtener_recompensa(estado):
    """Define las recompensas del entorno"""
    if estado == META:
        return 100  # Gran recompensa por llegar a la meta
    elif estado in OBSTACULOS:
        return -100  # Penalización por chocar con pared
    else:
        # Pequeña penalización para incentivar caminos más cortos
        return -1

def obtener_siguiente_estado(estado, accion):
    """Calcula el siguiente estado dada una acción"""
    i, j = estado
    if accion == 'arriba':
        i = max(0, i - 1)
    elif accion == 'abajo':
        i = min(FILAS - 1, i + 1)
    elif accion == 'izquierda':
        j = max(0, j - 1)
    elif accion == 'derecha':
        j = min(COLUMNAS - 1, j + 1)
    
    nuevo_estado = (i, j)
    if nuevo_estado in OBSTACULOS:
        return estado
    return nuevo_estado

def elegir_accion(estado, epsilon):
    """Política epsilon-greedy"""
    if np.random.random() < epsilon:
        return np.random.choice(['arriba', 'abajo', 'izquierda', 'derecha'])
    else:
        return max(Q_table[estado], key=Q_table[estado].get)

def entrenar_agente(episodios=500, learning_rate=0.1, gamma=0.95, epsilon_inicial=1.0):
    """Entrena el agente usando Q-Learning en el laberinto"""
    global Q_table, OBSTACULOS
    
    # Generar nuevo laberinto aleatorio
    densidad = random.uniform(0.25, 0.35)  # Densidad aleatoria entre 25% y 35%
    generar_laberinto_aleatorio(FILAS, COLUMNAS, densidad)
    
    inicializar_q_table()
    
    recompensas_por_episodio = []
    epsilon_decay = 0.995
    epsilon_min = 0.01
    epsilon = epsilon_inicial
    
    episodios_exitosos = 0
    
    for episodio in range(episodios):
        estado = INICIO
        recompensa_total = 0
        pasos = 0
        max_pasos = 200  # Más pasos para laberintos más grandes
        
        while estado != META and pasos < max_pasos:
            accion = elegir_accion(estado, epsilon)
            siguiente_estado = obtener_siguiente_estado(estado, accion)
            recompensa = obtener_recompensa(siguiente_estado)
            
            # Actualización Q-Learning
            mejor_valor_futuro = max(Q_table[siguiente_estado].values()) if siguiente_estado != META else 0
            Q_table[estado][accion] += learning_rate * (
                recompensa + gamma * mejor_valor_futuro - Q_table[estado][accion]
            )
            
            estado = siguiente_estado
            recompensa_total += recompensa
            pasos += 1
            
            # Salir si llegamos a la meta
            if estado == META:
                episodios_exitosos += 1
                break
        
        recompensas_por_episodio.append(recompensa_total)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # Guardar gráfico de recompensas
    graficar_recompensas(recompensas_por_episodio)
    
    return {
        'episodios': episodios,
        'recompensa_promedio': np.mean(recompensas_por_episodio[-100:]),
        'epsilon_final': epsilon,
        'recompensa_maxima': max(recompensas_por_episodio),
        'episodios_exitosos': episodios_exitosos,
        'tasa_exito': (episodios_exitosos / episodios) * 100,
        'obstaculos_totales': len(OBSTACULOS),
        'densidad_laberinto': len(OBSTACULOS) / (FILAS * COLUMNAS) * 100
    }

def graficar_recompensas(recompensas):
    """Genera gráfico mejorado de evolución de recompensas"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gráfico 1: Recompensas por episodio
    ax1.plot(recompensas, alpha=0.4, linewidth=0.8, color='steelblue', label='Recompensa por episodio')
    
    # Media móvil
    ventana = 50
    if len(recompensas) >= ventana:
        media_movil = np.convolve(recompensas, np.ones(ventana)/ventana, mode='valid')
        ax1.plot(range(ventana-1, len(recompensas)), media_movil, 'r-', 
                linewidth=2.5, label=f'Media Móvil ({ventana} episodios)')
    
    ax1.set_xlabel('Episodio', fontsize=11)
    ax1.set_ylabel('Recompensa Total', fontsize=11)
    ax1.set_title('Evolución de Recompensas durante el Entrenamiento del Laberinto', 
                 fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Gráfico 2: Histograma de recompensas
    ax2.hist(recompensas, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(recompensas), color='red', linestyle='--', linewidth=2, 
               label=f'Media: {np.mean(recompensas):.2f}')
    ax2.axvline(np.median(recompensas), color='green', linestyle='--', linewidth=2, 
               label=f'Mediana: {np.median(recompensas):.2f}')
    ax2.set_xlabel('Recompensa', fontsize=11)
    ax2.set_ylabel('Frecuencia', fontsize=11)
    ax2.set_title('Distribución de Recompensas', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, 'static', 'images', 'recompensas.png')
    plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()

def probar_agente():
    """Prueba el agente entrenado y visualiza la trayectoria en el laberinto"""
    estado = INICIO
    trayectoria = [estado]
    pasos = 0
    max_pasos = 100
    
    while estado != META and pasos < max_pasos:
        # Usar política greedy (sin exploración)
        accion = max(Q_table[estado], key=Q_table[estado].get)
        estado = obtener_siguiente_estado(estado, accion)
        trayectoria.append(estado)
        pasos += 1
        
        # Evitar ciclos infinitos
        if len(trayectoria) > 3 and trayectoria[-1] == trayectoria[-3]:
            print("Advertencia: Posible ciclo detectado")
            break
    
    visualizar_trayectoria(trayectoria)
    
    return {
        'trayectoria': trayectoria,
        'pasos': len(trayectoria) - 1,
        'exito': trayectoria[-1] == META,
        'estados_visitados': len(set(trayectoria))
    }

def visualizar_trayectoria(trayectoria):
    """Genera visualización mejorada de la trayectoria del agente en el laberinto"""
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Establecer límites
    ax.set_xlim(0, COLUMNAS)
    ax.set_ylim(0, FILAS)
    ax.set_aspect('equal')
    
    # Dibujar grid
    for i in range(FILAS + 1):
        ax.plot([0, COLUMNAS], [i, i], 'k-', linewidth=0.5, alpha=0.3)
    for j in range(COLUMNAS + 1):
        ax.plot([j, j], [0, FILAS], 'k-', linewidth=0.5, alpha=0.3)
    
    # Dibujar fondo del laberinto (caminos libres)
    for i in range(FILAS):
        for j in range(COLUMNAS):
            if (i, j) not in OBSTACULOS and (i, j) != INICIO and (i, j) != META:
                rect = patches.Rectangle((j, FILAS - i - 1), 1, 1, 
                                        linewidth=0, edgecolor='none', 
                                        facecolor='#f0f0f0', alpha=0.5)
                ax.add_patch(rect)
    
    # Dibujar obstáculos (paredes del laberinto) con colores variados
    colores_paredes = ['#2c3e50', '#34495e', '#1a252f']
    for idx, obstaculo in enumerate(OBSTACULOS):
        color = colores_paredes[idx % len(colores_paredes)]
        rect = patches.Rectangle((obstaculo[1], FILAS - obstaculo[0] - 1), 1, 1, 
                                linewidth=1, edgecolor='#000', 
                                facecolor=color, alpha=0.9)
        ax.add_patch(rect)
        # Agregar textura a las paredes
        ax.plot([obstaculo[1] + 0.1, obstaculo[1] + 0.9], 
               [FILAS - obstaculo[0] - 0.5, FILAS - obstaculo[0] - 0.5], 
               'w-', linewidth=0.5, alpha=0.2)
    
    # Marcar inicio (verde brillante con ícono)
    inicio_rect = patches.Rectangle((INICIO[1], FILAS - INICIO[0] - 1), 1, 1, 
                                   linewidth=3, edgecolor='#27ae60', 
                                   facecolor='#2ecc71', alpha=0.8)
    ax.add_patch(inicio_rect)
    ax.text(INICIO[1] + 0.5, FILAS - INICIO[0] - 0.5, '🏁', 
           ha='center', va='center', fontsize=20)
    ax.text(INICIO[1] + 0.5, FILAS - INICIO[0] - 0.8, 'INICIO', 
           ha='center', va='top', fontsize=8, fontweight='bold', color='white')
    
    # Marcar meta (rojo brillante con ícono)
    meta_rect = patches.Rectangle((META[1], FILAS - META[0] - 1), 1, 1, 
                                 linewidth=3, edgecolor='#c0392b', 
                                 facecolor='#e74c3c', alpha=0.8)
    ax.add_patch(meta_rect)
    ax.text(META[1] + 0.5, FILAS - META[0] - 0.5, '🎯', 
           ha='center', va='center', fontsize=20)
    ax.text(META[1] + 0.5, FILAS - META[0] - 0.2, 'META', 
           ha='center', va='bottom', fontsize=8, fontweight='bold', color='white')
    
    # Dibujar trayectoria con gradiente mejorado
    if len(trayectoria) > 1:
        tray_y = [FILAS - estado[0] - 0.5 for estado in trayectoria]
        tray_x = [estado[1] + 0.5 for estado in trayectoria]
        
        # Línea de trayectoria con gradiente de color
        for i in range(len(tray_x) - 1):
            color_intensity = i / len(tray_x)
            color = plt.cm.Blues(0.4 + color_intensity * 0.6)
            ax.plot([tray_x[i], tray_x[i+1]], [tray_y[i], tray_y[i+1]], 
                   color=color, linewidth=3, alpha=0.8)
        
        # Puntos con gradiente de tamaño y color
        for i, (x, y) in enumerate(zip(tray_x, tray_y)):
            alpha = 0.3 + (i / len(tray_x)) * 0.7
            size = 80 + (i / len(tray_x)) * 150
            color_val = 0.3 + (i / len(tray_x)) * 0.7
            ax.scatter(x, y, s=size, c=[plt.cm.Blues(color_val)], 
                      alpha=alpha, edgecolors='darkblue', linewidth=2, zorder=10)
        
        # Flechas direccionales cada ciertos pasos
        step_arrows = max(1, len(trayectoria) // 8)
        for i in range(step_arrows, len(trayectoria) - 1, step_arrows):
            ax.annotate('', xy=(tray_x[i+1], tray_y[i+1]), 
                       xytext=(tray_x[i], tray_y[i]),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.6))
    
    # Información del laberinto
    pasos = len(trayectoria) - 1
    exito = trayectoria[-1] == META if trayectoria else False
    densidad = (len(OBSTACULOS) / (FILAS * COLUMNAS)) * 100
    
    titulo = f'🧩 Laberinto Aleatorio {FILAS}x{COLUMNAS} - Trayectoria del Agente IA\n'
    if exito:
        titulo += f'✅ ¡RESUELTO! - Pasos: {pasos} | Obstáculos: {len(OBSTACULOS)} ({densidad:.1f}% densidad)'
    else:
        titulo += f'⏳ En progreso - Pasos: {pasos} | Obstáculos: {len(OBSTACULOS)} ({densidad:.1f}% densidad)'
    
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Agregar leyenda mejorada
    legend_text = (f'📍 Inicio: ({INICIO[0]}, {INICIO[1]})\n'
                  f'🎯 Meta: ({META[0]}, {META[1]})\n'
                  f'🧱 Obstáculos: {len(OBSTACULOS)}\n'
                  f'📊 Densidad: {densidad:.1f}%\n'
                  f'🔵 Camino: {pasos} pasos')
    
    ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='blue', linewidth=2))
    
    # Marca de dificultad
    if densidad < 25:
        dificultad = "⭐ FÁCIL"
        color_dif = '#2ecc71'
    elif densidad < 30:
        dificultad = "⭐⭐ MEDIO"
        color_dif = '#f39c12'
    else:
        dificultad = "⭐⭐⭐ DIFÍCIL"
        color_dif = '#e74c3c'
    
    ax.text(0.98, 0.98, dificultad, transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           fontweight='bold', color='white',
           bbox=dict(boxstyle='round', facecolor=color_dif, alpha=0.9, linewidth=2))
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(base_dir, 'static', 'images', 'trayectoria.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def reiniciar_modelo():
    """Reinicia el modelo Q-Learning y genera un nuevo laberinto"""
    global Q_table, OBSTACULOS
    # Generar nuevo laberinto aleatorio
    densidad = random.uniform(0.25, 0.35)
    generar_laberinto_aleatorio(FILAS, COLUMNAS, densidad)
    inicializar_q_table()
    return {
        'obstaculos': len(OBSTACULOS),
        'densidad': (len(OBSTACULOS) / (FILAS * COLUMNAS)) * 100
    }

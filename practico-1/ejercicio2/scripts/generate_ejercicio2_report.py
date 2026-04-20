#!/usr/bin/env python3
"""
Script para generar gráficas e informe del Ejercicio 2
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Directorios
RESULTS_DIR = '../results'
PLOTS_DIR = '../results/plots'
REPORTS_DIR = '../results/reports'

# Crear directorios si no existen
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def load_data():
    """Carga todos los archivos CSV de resultados"""
    data = {}
    
    # Block sizes
    bs_file = f'{RESULTS_DIR}/ejercicio2_block_sizes.csv'
    if os.path.exists(bs_file):
        data['block_sizes'] = pd.read_csv(bs_file)
    
    # Blocked vs IKJ
    comp_file = f'{RESULTS_DIR}/ejercicio2_blocked_vs_ikj.csv'
    if os.path.exists(comp_file):
        data['blocked_vs_ikj'] = pd.read_csv(comp_file)
    
    # Loop orders
    loop_file = f'{RESULTS_DIR}/ejercicio2_loop_orders.csv'
    if os.path.exists(loop_file):
        data['loop_orders'] = pd.read_csv(loop_file)
    
    return data

def plot_block_sizes(df):
    """Gráfica de rendimiento vs tamaño de bloque"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tiempo vs BS
    ax1.plot(df['block_size'], df['tiempo_segundos'], 'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Tamaño de Bloque (BS)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tiempo (segundos)', fontsize=12, fontweight='bold')
    ax1.set_title('Tiempo de Ejecución vs Tamaño de Bloque', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # Marcar el óptimo
    optimal_idx = df['tiempo_segundos'].idxmin()
    optimal_bs = df.loc[optimal_idx, 'block_size']
    optimal_time = df.loc[optimal_idx, 'tiempo_segundos']
    ax1.plot(optimal_bs, optimal_time, 'r*', markersize=20, label=f'Óptimo: BS={optimal_bs}')
    ax1.legend(fontsize=11)
    
    # MFLOPS vs BS
    ax2.plot(df['block_size'], df['mflops'], 'o-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Tamaño de Bloque (BS)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MFLOPS', fontsize=12, fontweight='bold')
    ax2.set_title('Rendimiento (MFLOPS) vs Tamaño de Bloque', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # Marcar el óptimo
    optimal_mflops = df.loc[optimal_idx, 'mflops']
    ax2.plot(optimal_bs, optimal_mflops, 'r*', markersize=20, label=f'Óptimo: {optimal_mflops:.0f} MFLOPS')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/block_sizes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {PLOTS_DIR}/block_sizes.png")

def plot_blocked_vs_ikj(df):
    """Gráfica comparativa Blocked vs IKJ"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extraer datos
    variants = df['variante'].tolist()
    times = df['tiempo_segundos'].tolist()
    mflops = df['mflops'].tolist()
    block_sizes = df['block_size'].tolist()
    
    # Gráfica de tiempos
    colors = ['#2E86AB', '#A23B72']
    bars1 = ax1.bar(range(len(variants)), times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels([v.upper() if v != 'blocked' else f'Blocked\n(BS={block_sizes[i]})' for i, v in enumerate(variants)], fontsize=11)
    ax1.set_ylabel('Tiempo (segundos)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparación de Tiempo: Blocked vs IKJ', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars1, times)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Gráfica de MFLOPS
    bars2 = ax2.bar(range(len(variants)), mflops, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels([v.upper() if v != 'blocked' else f'Blocked\n(BS={block_sizes[i]})' for i, v in enumerate(variants)], fontsize=11)
    ax2.set_ylabel('MFLOPS', fontsize=12, fontweight='bold')
    ax2.set_title('Comparación de Rendimiento: Blocked vs IKJ', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for i, (bar, val) in enumerate(zip(bars2, mflops)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Calcular speedup
    if len(times) == 2:
        speedup = times[0] / times[1]
        fig.suptitle(f'Speedup: {speedup:.2f}x (IKJ es {speedup:.2f}x más rápido)', 
                    fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/blocked_vs_ikj.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {PLOTS_DIR}/blocked_vs_ikj.png")

def plot_loop_orders(df):
    """Gráficas de comparación de órdenes de loop"""
    sizes = df['size'].unique()
    variants = df['variant'].unique()
    
    # Gráfica 1: MFLOPS por variante para cada tamaño
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(range(len(variants)))
    
    for idx, size in enumerate(sizes):
        ax = axes[idx]
        df_size = df[df['size'] == size]
        
        bars = ax.bar(range(len(variants)), df_size['mflops'], 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(df_size['variant'].str.upper(), fontsize=10)
        ax.set_ylabel('MFLOPS', fontsize=11, fontweight='bold')
        ax.set_title(f'Tamaño: {size}×{size}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Marcar el mejor
        best_idx = df_size['mflops'].idxmax()
        best_variant = df_size.loc[best_idx, 'variant']
        best_mflops = df_size.loc[best_idx, 'mflops']
        
        # Añadir valores sobre las barras más altas
        for i, (bar, val) in enumerate(zip(bars, df_size['mflops'])):
            if val > best_mflops * 0.5:  # Solo mostrar valores altos
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Rendimiento por Orden de Loop y Tamaño de Matriz', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/loop_orders_by_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {PLOTS_DIR}/loop_orders_by_size.png")
    
    # Gráfica 2: Comparación de variantes a través de tamaños
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for variant in variants:
        df_variant = df[df['variant'] == variant]
        ax.plot(df_variant['size'], df_variant['mflops'], 
               'o-', linewidth=2, markersize=8, label=variant.upper())
    
    ax.set_xlabel('Tamaño de Matriz (N×N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('MFLOPS', fontsize=12, fontweight='bold')
    ax.set_title('Rendimiento de Variantes vs Tamaño de Matriz', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/loop_orders_scaling.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {PLOTS_DIR}/loop_orders_scaling.png")
    
    # Gráfica 3: Heatmap de rendimiento
    pivot_df = df.pivot(index='variant', columns='size', values='mflops')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes, fontsize=11)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(pivot_df.index.str.upper(), fontsize=11)
    
    ax.set_xlabel('Tamaño de Matriz', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variante de Loop', fontsize=12, fontweight='bold')
    ax.set_title('Mapa de Calor: MFLOPS por Variante y Tamaño', fontsize=14, fontweight='bold')
    
    # Añadir valores en las celdas
    for i in range(len(variants)):
        for j in range(len(sizes)):
            text = ax.text(j, i, f'{pivot_df.values[i, j]:.0f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('MFLOPS', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/loop_orders_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Gráfica guardada: {PLOTS_DIR}/loop_orders_heatmap.png")

def generate_markdown_report(data):
    """Genera un informe en Markdown con análisis completo"""
    
    report = f"""# Informe del Ejercicio 2: Multiplicación de Matrices

**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Resumen Ejecutivo

Este informe presenta los resultados del análisis de rendimiento de diferentes técnicas de multiplicación de matrices, enfocándose en el impacto del **blocking** y el **orden de loops** en el uso eficiente de la memoria caché.

---

## Parte 1: Determinación del Tamaño de Bloque Óptimo

### Resultados

"""
    
    if 'block_sizes' in data:
        df_bs = data['block_sizes']
        optimal_idx = df_bs['tiempo_segundos'].idxmin()
        optimal_bs = df_bs.loc[optimal_idx, 'block_size']
        optimal_time = df_bs.loc[optimal_idx, 'tiempo_segundos']
        optimal_mflops = df_bs.loc[optimal_idx, 'mflops']
        
        report += f"""
| BS | Tiempo (s) | MFLOPS |
|----|-----------|--------|
"""
        for _, row in df_bs.iterrows():
            marker = " ⭐ **ÓPTIMO**" if row['block_size'] == optimal_bs else ""
            report += f"| {row['block_size']} | {row['tiempo_segundos']:.4f} | {row['mflops']:.2f}{marker} |\n"
        
        report += f"""
### Análisis

- **BS óptimo detectado:** {optimal_bs}
- **Tiempo:** {optimal_time:.4f} segundos
- **Rendimiento:** {optimal_mflops:.2f} MFLOPS

#### Observaciones:

1. **BS pequeños (8-16):** Mejor rendimiento debido a que los bloques caben completamente en caché L1
2. **BS medianos (32-64):** Rendimiento intermedio, bloques más grandes que L1 pero menores que L2
3. **BS grandes (128-256):** Peor rendimiento por *cache thrashing* y menor reutilización de datos

![Block Sizes](../plots/block_sizes.png)

---

"""
    
    if 'blocked_vs_ikj' in data:
        df_comp = data['blocked_vs_ikj']
        
        # Extraer datos (asumiendo que blocked es la primera fila)
        blocked_time = df_comp.loc[0, 'tiempo_segundos']
        blocked_mflops = df_comp.loc[0, 'mflops']
        ikj_time = df_comp.loc[1, 'tiempo_segundos']
        ikj_mflops = df_comp.loc[1, 'mflops']
        speedup = blocked_time / ikj_time
        
        report += f"""## Parte 2: Comparación Blocked vs IKJ

### Resultados

| Variante | Tiempo (s) | MFLOPS | Speedup |
|----------|-----------|--------|---------|
| Blocked (BS={optimal_bs}) | {blocked_time:.4f} | {blocked_mflops:.2f} | 1.00x |
| IKJ | {ikj_time:.4f} | {ikj_mflops:.2f} | **{speedup:.2f}x** |

### Análisis

- **IKJ es {speedup:.2f}x más rápido** que Blocked con BS={optimal_bs}
- **Razón principal:** IKJ tiene acceso secuencial a todas las matrices, mientras que Blocked aún accede a B por columnas

#### ¿Por qué IKJ supera a Blocking?

**Acceso a memoria en IKJ:**
```
for i:          // Fija fila de A y C
    for k:      // Fija columna de A, fila de B
        for j:  // Recorre secuencialmente
            C[i][j] += A[i][k] * B[k][j]
            //  ↑          ↑         ↑
            // Secuencial  Reuso   Secuencial
```

**Problema del Blocking implementado:**
- Divide en bloques pero no optimiza el acceso por columnas a B
- Overhead adicional de los loops externos de blocking
- BS={optimal_bs} no es suficientemente pequeño para evitar conflictos

![Blocked vs IKJ](../plots/blocked_vs_ikj.png)

---

"""
    
    if 'loop_orders' in data:
        df_loop = data['loop_orders']
        
        report += """## Parte 3: Comparación de Órdenes de Loop

### Resultados por Tamaño de Matriz

"""
        
        sizes = df_loop['size'].unique()
        variants = df_loop['variant'].unique()
        
        for size in sizes:
            df_size = df_loop[df_loop['size'] == size].sort_values('mflops', ascending=False)
            report += f"\n#### Matriz {size}×{size}\n\n"
            report += "| Variante | Tiempo (s) | MFLOPS | Ranking |\n"
            report += "|----------|-----------|--------|----------|\n"
            
            for rank, (_, row) in enumerate(df_size.iterrows(), 1):
                medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
                report += f"| {row['variant'].upper()} | {row['tiempo_segundos']:.4f} | {row['mflops']:.2f} | {rank} {medal} |\n"
        
        report += """
### Análisis General

#### Ranking de Variantes (promedio):

"""
        
        avg_perf = df_loop.groupby('variant')['mflops'].mean().sort_values(ascending=False)
        for rank, (variant, mflops) in enumerate(avg_perf.items(), 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "")
            report += f"{rank}. **{variant.upper()}** {medal}: {mflops:.2f} MFLOPS promedio\n"
        
        report += """
#### Explicación de Rendimiento:

**🥇 IKJ (El Campeón):**
- Acceso secuencial a A, B y C
- Perfecta localidad espacial
- Máximo aprovechamiento de cache lines

**🥈 KIJ (Segundo Lugar):**
- Similar a IKJ pero diferente orden
- Buen acceso secuencial
- Ligeramente menos eficiente por patrón de escritura

**🥉 IJK/JIK (Intermedio):**
- Acceso por columnas a B
- Causa cache misses frecuentes
- ~10x más lento que IKJ

**🐌 JKI/KJI (Los Peores):**
- Doble acceso por columnas (A y C)
- Máximos cache misses
- ~50-70x más lento que IKJ

#### Impacto del Tamaño de Matriz:

**Potencias de 2 (256, 512, 1024):**
- Peor rendimiento en JKI/KJI (~200-300 MFLOPS)
- Causa: Conflictos de mapeo en caché asociativa por conjuntos

**No potencias de 2 (260, 550, 1050):**
- Mejor rendimiento en JKI/KJI (~1500-2700 MFLOPS)
- Causa: Desalineación reduce conflictos de caché

![Loop Orders by Size](../plots/loop_orders_by_size.png)

![Loop Orders Scaling](../plots/loop_orders_scaling.png)

![Loop Orders Heatmap](../plots/loop_orders_heatmap.png)

---

## Conclusiones

### 1. El Orden de Loop es Crítico
- La diferencia entre el mejor (IKJ) y el peor (JKI/KJI) es de **~70x**
- Un simple cambio en el orden de loops puede multiplicar el rendimiento

### 2. Localidad de Datos > Complejidad Algorítmica
- IKJ simple supera a Blocking complejo
- La clave es el acceso secuencial a memoria

### 3. El Tamaño de Bloque Importa
- BS óptimo: {optimal_bs}
- Bloques demasiado grandes causan cache thrashing
- Bloques demasiado pequeños aumentan overhead

### 4. Tamaños de Matriz y Conflictos de Caché
- Potencias de 2 pueden causar conflictos en caché asociativa
- Usar tamaños no potencias de 2 puede mejorar rendimiento

### 5. Lección Principal
**El rendimiento en computación de alto rendimiento depende más de cómo se accede a memoria que del algoritmo matemático en sí.**

---

## Recomendaciones

1. **Siempre usar orden IKJ o KIJ** para multiplicación de matrices
2. **Evitar órdenes con acceso por columnas** (IJK, JIK, JKI, KJI)
3. **Ajustar BS según el tamaño de caché** del procesador
4. **Considerar tamaños de matriz** que eviten conflictos de caché
5. **Medir siempre** antes de optimizar - los resultados pueden sorprender

---

*Informe generado automáticamente por `generate_ejercicio2_report.py`*
"""
    
    # Guardar informe
    report_file = f'{REPORTS_DIR}/INFORME_EJERCICIO2.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Informe guardado: {report_file}")

def main():
    print("=" * 60)
    print("  Generador de Gráficas e Informe - Ejercicio 2")
    print("=" * 60)
    print()
    
    # Cargar datos
    print("📊 Cargando datos...")
    data = load_data()
    
    if not data:
        print("❌ Error: No se encontraron archivos de resultados")
        print("   Ejecuta primero: make run2")
        return
    
    print(f"✓ Datos cargados: {list(data.keys())}")
    print()
    
    # Generar gráficas
    print("📈 Generando gráficas...")
    
    if 'block_sizes' in data:
        plot_block_sizes(data['block_sizes'])
    
    if 'blocked_vs_ikj' in data:
        plot_blocked_vs_ikj(data['blocked_vs_ikj'])
    
    if 'loop_orders' in data:
        plot_loop_orders(data['loop_orders'])
    
    print()
    
    # Generar informe
    print("📝 Generando informe...")
    generate_markdown_report(data)
    print()
    
    print("=" * 60)
    print("✅ Proceso completado exitosamente")
    print("=" * 60)
    print()
    print(f"📁 Gráficas guardadas en: {PLOTS_DIR}/")
    print(f"📄 Informe guardado en: {REPORTS_DIR}/INFORME_EJERCICIO2.md")
    print()

if __name__ == '__main__':
    main()

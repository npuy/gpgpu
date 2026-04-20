#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

df_results = pd.read_csv('../results/ejercicio1_results.csv')
df_detailed = pd.read_csv('../results/ejercicio1_detailed.csv')
df_strides = pd.read_csv('../results/ejercicio1_strides.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Ejercicio 1: Análisis de Localidad Espacial y Temporal', fontsize=16, fontweight='bold')

ax1 = axes[0, 0]
iterations = df_detailed['iteracion']
seq_times = df_detailed['secuencial']
rand_times = df_detailed['aleatorio']

ax1.plot(iterations, seq_times, marker='o', linewidth=2, markersize=8, label='Secuencial', color='#2ecc71')
ax1.plot(iterations, rand_times, marker='s', linewidth=2, markersize=8, label='Aleatorio', color='#e74c3c')
ax1.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, label='Caché se calienta')
ax1.set_xlabel('Iteración', fontweight='bold')
ax1.set_ylabel('Tiempo (segundos)', fontweight='bold')
ax1.set_title('Acceso Secuencial vs Aleatorio\n(Efecto de Caché Fría/Caliente)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
categories = ['Secuencial\nFría', 'Secuencial\nCaliente', 'Aleatorio\nFría', 'Aleatorio\nCaliente']
times = [
    df_results[df_results['tipo'] == 'secuencial_fria']['tiempo_segundos'].values[0],
    df_results[df_results['tipo'] == 'secuencial_caliente']['tiempo_segundos'].values[0],
    df_results[df_results['tipo'] == 'aleatorio_fria']['tiempo_segundos'].values[0],
    df_results[df_results['tipo'] == 'aleatorio_caliente']['tiempo_segundos'].values[0]
]
colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
bars = ax2.bar(categories, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for i, (bar, time) in enumerate(zip(bars, times)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s',
             ha='center', va='bottom', fontweight='bold', fontsize=10)

ax2.set_ylabel('Tiempo (segundos)', fontweight='bold')
ax2.set_title('Comparación de Tiempos\n(Caché Fría vs Caliente)', fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)

ax3 = axes[1, 0]
stride_values = df_strides['stride'].unique()
stride_avg_times = []

for stride in stride_values:
    stride_data = df_strides[df_strides['stride'] == stride]
    warm_times = stride_data[stride_data['iteracion'] > 1]['tiempo_segundos']
    stride_avg_times.append(warm_times.mean())

ax3.plot(stride_values, stride_avg_times, marker='o', linewidth=2.5, markersize=10, 
         color='#9b59b6', markerfacecolor='#e74c3c', markeredgewidth=2, markeredgecolor='#9b59b6')
ax3.axhline(y=stride_avg_times[0], color='green', linestyle='--', alpha=0.5, label='Baseline (stride=1)')
ax3.axvline(x=64, color='orange', linestyle='--', alpha=0.5, label='Tamaño línea caché (64B)')
ax3.set_xscale('log', base=2)
ax3.set_xlabel('Stride (bytes, escala log₂)', fontweight='bold')
ax3.set_ylabel('Tiempo promedio (segundos)', fontweight='bold')
ax3.set_title('Impacto del Stride en el Rendimiento\n(Caché Caliente)', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')
ax3.set_xticks(stride_values)
ax3.set_xticklabels(stride_values)

ax4 = axes[1, 1]
seq_baseline = stride_avg_times[0]
slowdowns = [t / seq_baseline for t in stride_avg_times]

colors_stride = ['#2ecc71' if s < 1.5 else '#f39c12' if s < 3.0 else '#e67e22' if s < 10.0 else '#e74c3c' 
                 for s in slowdowns]
bars = ax4.bar(range(len(stride_values)), slowdowns, color=colors_stride, alpha=0.8, 
               edgecolor='black', linewidth=1.5)

for i, (bar, slowdown) in enumerate(zip(bars, slowdowns)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{slowdown:.1f}x',
             ha='center', va='bottom', fontweight='bold', fontsize=9)

ax4.set_xticks(range(len(stride_values)))
ax4.set_xticklabels(stride_values)
ax4.set_xlabel('Stride (bytes)', fontweight='bold')
ax4.set_ylabel('Slowdown vs Stride=1', fontweight='bold')
ax4.set_title('Degradación de Rendimiento por Stride\n(Verde=Excelente, Amarillo=Bueno, Naranja=Regular, Rojo=Pobre)', 
              fontweight='bold', fontsize=10)
ax4.axhline(y=1, color='black', linestyle='-', linewidth=1, alpha=0.3)
ax4.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('ejercicio1_analisis.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: ejercicio1_analisis.png")

fig2, ax = plt.subplots(figsize=(12, 7))
for stride in stride_values:
    stride_data = df_strides[df_strides['stride'] == stride]
    iterations = stride_data['iteracion']
    times = stride_data['tiempo_segundos']
    ax.plot(iterations, times, marker='o', linewidth=2, markersize=7, label=f'Stride {stride}')

ax.set_xlabel('Iteración', fontweight='bold')
ax.set_ylabel('Tiempo (segundos)', fontweight='bold')
ax.set_title('Evolución Temporal por Stride\n(Efecto de Calentamiento de Caché)', fontweight='bold', fontsize=14)
ax.legend(loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
ax.text(1.5, ax.get_ylim()[1]*0.9, 'Caché se calienta →', ha='left', fontsize=10, style='italic')

plt.tight_layout()
plt.savefig('ejercicio1_strides_temporal.png', dpi=300, bbox_inches='tight')
print("Gráfico guardado: ejercicio1_strides_temporal.png")

print("\n✓ Todos los gráficos generados exitosamente")

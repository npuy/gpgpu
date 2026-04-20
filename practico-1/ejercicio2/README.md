# Ejercicio 2: Multiplicación de Matrices

Análisis del impacto del blocking y el orden de loops en la multiplicación de matrices.

## Estructura

```
ejercicio2/
├── src/
│   ├── ejercicio2.c    # Código fuente principal
│   └── utils.h         # Utilidades (medición de tiempo)
├── results/
│   └── (archivos CSV generados por el programa)
└── Makefile
```

## Compilar y Ejecutar

```bash
# Compilar
make

# Ejecutar
make run

# Limpiar
make clean
```

## Desde la raíz del proyecto

```bash
# Compilar ejercicio 2
make ejercicio2

# Ejecutar ejercicio 2
make run2

# Limpiar ejercicio 2
make clean2
```

## Pruebas Realizadas

1. **Determinación del BS óptimo**: Prueba diferentes tamaños de bloque
2. **Comparación Blocked vs IKJ**: Compara blocking con orden IKJ
3. **Comparación de órdenes de loop**: Evalúa las 6 permutaciones (ijk, jik, ikj, kij, jki, kji)

## Resultados

Los resultados se guardan en `results/`:
- `ejercicio2_block_sizes.csv`
- `ejercicio2_blocked_vs_ikj.csv`
- `ejercicio2_loop_orders.csv`

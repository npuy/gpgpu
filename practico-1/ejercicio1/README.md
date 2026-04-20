# Ejercicio 1: Localidad Espacial y Temporal

Análisis del impacto de diferentes patrones de acceso a memoria en el rendimiento de la caché CPU.

## Estructura

```
ejercicio1/
├── src/
│   ├── ejercicio1.c    # Código fuente principal
│   └── utils.h         # Utilidades (medición de tiempo)
├── results/
│   ├── ejercicio1_results.csv       # Resumen de tiempos
│   ├── ejercicio1_detailed.csv      # Tiempos por iteración
│   ├── ejercicio1_strides.csv       # Datos de stride
│   ├── INFORME_EJERCICIO1.md        # Informe completo
│   └── RESUMEN_EJERCICIO1.txt       # Resumen ejecutivo
├── scripts/
│   └── generate_plots.py            # Generación de gráficos
└── Makefile
```

## Compilar y Ejecutar

```bash
# Compilar
make

# Ejecutar
make run

# Generar gráficos (requiere matplotlib y pandas)
make plots

# Limpiar
make clean
```

## Desde la raíz del proyecto

```bash
# Compilar ejercicio 1
make ejercicio1

# Ejecutar ejercicio 1
make run1

# Generar gráficos ejercicio 1
make plots1

# Limpiar ejercicio 1
make clean1
```

## Resultados

Los resultados se guardan automáticamente en `results/`:
- **CSV files**: Datos crudos para análisis
- **INFORME_EJERCICIO1.md**: Análisis completo con conclusiones
- **RESUMEN_EJERCICIO1.txt**: Resumen de hallazgos clave

## Gráficos

Para generar visualizaciones:

```bash
cd scripts
python3 -m venv venv
source venv/bin/activate
pip install matplotlib pandas
python3 generate_plots.py
```

Genera:
- `ejercicio1_analisis.png`: Panel con 4 gráficos comparativos
- `ejercicio1_strides_temporal.png`: Evolución temporal de strides

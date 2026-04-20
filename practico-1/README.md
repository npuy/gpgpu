# Práctico: Análisis de Memoria Caché

Este proyecto contiene ejercicios prácticos para analizar el comportamiento de la memoria caché en sistemas de computación.

## Estructura del Proyecto

```
practico-memoria-cache/
├── ejercicio1/              # Ejercicio 1: Localidad Espacial
│   ├── src/                 # Código fuente
│   ├── results/             # Resultados y análisis
│   ├── scripts/             # Scripts de visualización
│   ├── Makefile
│   └── README.md
├── ejercicio2/              # Ejercicio 2: Multiplicación de Matrices
│   ├── src/                 # Código fuente
│   ├── results/             # Resultados
│   ├── Makefile
│   └── README.md
├── scripts/                 # Scripts compartidos (si los hay)
├── Makefile                 # Makefile principal
└── README.md
```

## Compilación Rápida

### Compilar todo
```bash
make all
```

### Compilar ejercicios individuales
```bash
make ejercicio1
make ejercicio2
```

## Ejecución

### Ejecutar ejercicios
```bash
make run1    # Ejecuta ejercicio 1
make run2    # Ejecuta ejercicio 2
```

### Generar gráficos (Ejercicio 1)
```bash
make plots1
```

## Limpieza

```bash
make clean      # Limpia todo
make clean1     # Limpia solo ejercicio 1
make clean2     # Limpia solo ejercicio 2
```

## Ejercicios

### Ejercicio 1: Localidad Espacial y Temporal
Analiza el impacto de diferentes patrones de acceso a memoria:
- Acceso secuencial vs aleatorio
- Efecto de caché fría vs caliente
- Análisis de diferentes strides (1, 16, 64, 128, 256, 512, 1024, 4096 bytes)

**Resultados:** Ver `ejercicio1/results/INFORME_EJERCICIO1.md`

### Ejercicio 2: Multiplicación de Matrices
Estudia el efecto del blocking y el orden de loops:
- Determinación del tamaño de bloque óptimo
- Comparación de blocking vs orden IKJ
- Análisis de las 6 permutaciones de loops

**Resultados:** Ver `ejercicio2/results/`

## Trabajo Individual

Cada ejercicio es independiente y puede compilarse/ejecutarse por separado:

```bash
cd ejercicio1
make && make run

cd ../ejercicio2
make && make run
```

Consulta el README.md de cada ejercicio para más detalles.

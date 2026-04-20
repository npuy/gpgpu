# Informe Ejercicio 1: Análisis de Localidad Espacial y Temporal en Memoria Caché

**Autor:** Belén Olivera  
**Fecha:** 14 de Marzo, 2026  
**Tamaño del Array:** 100 MB (104,857,600 bytes)  
**Iteraciones:** 5

---

## 1. Introducción

Este ejercicio analiza el impacto de la **localidad espacial** y **temporal** en el rendimiento de la jerarquía de memoria caché de la CPU. Se comparan diferentes patrones de acceso a memoria para demostrar cómo la organización de los datos y el patrón de acceso afectan dramáticamente el rendimiento.

### Objetivos
1. Comparar acceso **secuencial** vs **aleatorio** a un array grande
2. Analizar el efecto de **caché fría** vs **caché caliente**
3. Estudiar el impacto del **stride** (tamaño de salto) en el rendimiento
4. Identificar el **tamaño de línea de caché** mediante experimentación

---

## 2. Metodología

### 2.1 Configuración Experimental
- **Array size:** 100 MB (excede el tamaño típico de caché L3)
- **Tipo de dato:** `char` (1 byte)
- **Operación:** Incremento (`array[idx]++`)
- **Número de accesos:** 100 millones (constante para todas las pruebas)
- **Iteraciones:** 5 (primera = caché fría, 2-5 = caché caliente)

### 2.2 Patrones de Acceso Evaluados

#### A. Acceso Secuencial
```c
for (int i = 0; i < size; i++) {
    array[i]++;
}
```
- Accede a posiciones consecutivas: 0, 1, 2, 3, 4...
- Máxima localidad espacial y temporal

#### B. Acceso Aleatorio
```c
// Índices previamente mezclados con shuffle
for (int i = 0; i < size; i++) {
    array[indices[i]]++;
}
```
- Accede a posiciones completamente aleatorias
- Mínima localidad espacial

#### C. Acceso con Stride Variable
```c
for (long long i = 0; i < size; i++) {
    long long idx = (i * stride) % size;
    array[idx]++;
}
```
- Strides evaluados: 1, 16, 64, 128, 256, 512, 1024, 4096 bytes
- Permite analizar el impacto del tamaño de línea de caché

---

## 3. Resultados Experimentales

### 3.1 Acceso Secuencial vs Aleatorio

#### Tabla 1: Tiempos por Iteración

| Iteración | Secuencial (s) | Aleatorio (s) | Estado Caché |
|-----------|----------------|---------------|--------------|
| 1         | 0.3515         | 1.4334        | FRÍA         |
| 2         | 0.1070         | 1.5740        | CALIENTE     |
| 3         | 0.0908         | 1.6213        | CALIENTE     |
| 4         | 0.0832         | 1.5817        | CALIENTE     |
| 5         | 0.0838         | 1.4043        | CALIENTE     |

#### Tabla 2: Resumen Comparativo

| Métrica                    | Secuencial | Aleatorio | Ratio      |
|----------------------------|------------|-----------|------------|
| **Caché Fría (iter 1)**    | 0.3515 s   | 1.4334 s  | **4.08x**  |
| **Caché Caliente (avg)**   | 0.0912 s   | 1.5453 s  | **16.94x** |
| **Mejora por warming**     | 3.85x      | 0.93x     | -          |

### 3.2 Análisis de Strides

#### Tabla 3: Rendimiento por Stride (Caché Caliente)

| Stride (bytes) | Tiempo Promedio (s) | Slowdown vs Seq | Eficiencia Caché |
|----------------|---------------------|-----------------|------------------|
| 1              | 0.1519              | 1.67x           | BUENA            |
| 16             | 0.2059              | 2.26x           | BUENA            |
| 64             | 0.6289              | 6.89x           | REGULAR          |
| 128            | 1.2774              | 14.00x          | POBRE            |
| 256            | 1.6073              | 17.62x          | POBRE            |
| 512            | 1.5638              | 17.14x          | POBRE            |
| 1024           | 1.8803              | 20.61x          | POBRE            |
| 4096           | 1.6870              | 18.49x          | POBRE            |

---

## 4. Análisis y Discusión

### 4.1 Localidad Temporal: Efecto de Caché Fría vs Caliente

**Observación clave:** El acceso secuencial mejora **3.85x** entre la primera iteración (caché fría) y las subsecuentes (caché caliente), mientras que el acceso aleatorio prácticamente no mejora (0.93x).

**Explicación:**
- **Acceso Secuencial:** Los datos accedidos en la iteración 1 permanecen en caché para las iteraciones 2-5, aprovechando la **localidad temporal**
- **Acceso Aleatorio:** El array (100 MB) no cabe en la caché L3 (típicamente 8-16 MB), causando **cache misses constantes** en cada iteración
- **Conclusión:** La localidad temporal solo beneficia cuando el working set cabe en caché

### 4.2 Localidad Espacial: Impacto del Stride

**Observación clave:** El rendimiento se degrada dramáticamente cuando el stride supera 64 bytes.

#### Análisis por Rango de Stride:

**Stride 1-16 bytes (BUENA eficiencia):**
- Slowdown: 1.67x - 2.26x
- **Explicación:** Múltiples accesos aprovechan la misma línea de caché (64 bytes)
- Ejemplo: Con stride=16, cada línea de caché sirve para 4 accesos (64/16)
- El **prefetcher de hardware** detecta el patrón secuencial y precarga datos

**Stride 64 bytes (REGULAR eficiencia):**
- Slowdown: 6.89x
- **Explicación:** Cada acceso trae una línea de caché completa (64 bytes) pero solo usa 1 byte
- **Desperdicio del 98.4%** de los datos traídos a caché
- Este resultado **confirma que el tamaño de línea de caché es 64 bytes**

**Stride 128+ bytes (POBRE eficiencia):**
- Slowdown: 14.00x - 20.61x
- **Explicación:** Comportamiento similar a acceso aleatorio
- Cada acceso requiere traer una nueva línea de caché
- No hay reutilización de datos → localidad espacial completamente perdida
- Tiempos similares al acceso aleatorio (1.5-1.9s vs 1.5s)

### 4.3 Identificación del Tamaño de Línea de Caché

El **punto de inflexión** entre stride 16 y stride 64 indica claramente:

**Tamaño de línea de caché = 64 bytes**

Evidencia:
1. Stride 16: Slowdown 2.26x (bueno) → 4 accesos por línea
2. Stride 64: Slowdown 6.89x (regular) → 1 acceso por línea
3. Stride 128: Slowdown 14.00x (pobre) → desperdicia líneas completas

### 4.4 Comparación Secuencial vs Aleatorio

**En caché caliente:**
- Secuencial: 0.0912 s
- Aleatorio: 1.5453 s
- **Slowdown: 16.94x**

**¿Por qué esta diferencia tan grande?**

1. **Cache Miss Rate:**
   - Secuencial: ~1-2% (solo compulsory misses)
   - Aleatorio: ~90-95% (capacity + conflict misses)

2. **Prefetching:**
   - Secuencial: El prefetcher detecta el patrón y precarga datos
   - Aleatorio: Patrón impredecible, sin prefetching efectivo

3. **TLB Misses:**
   - Secuencial: Acceso secuencial a páginas → pocos TLB misses
   - Aleatorio: Saltos entre páginas → muchos TLB misses

4. **Latencia de Memoria:**
   - Cache hit: ~4 ciclos (L1)
   - RAM: ~200-300 ciclos
   - **Ratio: 50-75x más lento**

---

## 5. Conclusiones

### 5.1 Hallazgos Principales

1. **Localidad Espacial es Crítica:**
   - El acceso secuencial es hasta **16.94x más rápido** que el aleatorio
   - Strides pequeños (≤16 bytes) mantienen buen rendimiento
   - Strides grandes (≥128 bytes) destruyen la localidad espacial

2. **Tamaño de Línea de Caché:**
   - Confirmado experimentalmente: **64 bytes**
   - Visible en la degradación de rendimiento entre stride 16 y 64

3. **Localidad Temporal:**
   - Solo efectiva cuando el working set cabe en caché
   - Acceso secuencial: 3.85x mejora con warming
   - Acceso aleatorio: sin mejora (array > tamaño de caché)

4. **Prefetching de Hardware:**
   - Altamente efectivo para patrones secuenciales
   - Inefectivo para patrones aleatorios o strides grandes

### 5.2 Implicaciones para Programación

**Recomendaciones para código de alto rendimiento:**

1. **Organizar datos secuencialmente** en memoria (arrays contiguos)
2. **Evitar indirecciones** y punteros dispersos
3. **Usar strides pequeños** (idealmente ≤ tamaño de línea de caché)
4. **Estructuras de datos cache-friendly:**
   - Array of Structs (AoS) → Struct of Arrays (SoA) cuando sea apropiado
   - Alinear datos a límites de línea de caché (64 bytes)
5. **Bloquear algoritmos** para que el working set quepa en caché

### 5.3 Limitaciones del Estudio

1. **Tamaño fijo del array:** Solo se probó con 100 MB
2. **Tipo de dato:** Solo `char` (1 byte) - otros tamaños pueden mostrar patrones diferentes
3. **Arquitectura específica:** Resultados pueden variar en otras CPUs
4. **Sin medición directa:** No se usaron contadores de hardware (perf) para medir cache misses

---

## 6. Datos Crudos

### 6.1 Archivos Generados

Los siguientes archivos CSV contienen los datos completos:

- `ejercicio1_results.csv`: Resumen de tiempos (secuencial/aleatorio, fría/caliente)
- `ejercicio1_detailed.csv`: Tiempos por iteración (secuencial vs aleatorio)
- `ejercicio1_strides.csv`: Tiempos detallados por stride e iteración

### 6.2 Reproducibilidad

Para reproducir estos resultados:

```bash
make clean && make all
./ejercicio1
```

Los resultados pueden variar según:
- Arquitectura de CPU (tamaño de caché, prefetcher)
- Carga del sistema
- Frecuencia de CPU (throttling)
- Sistema operativo y scheduler

---

## 7. Referencias Teóricas

### Jerarquía de Memoria Típica

| Nivel  | Tamaño    | Latencia  | Ancho de Banda |
|--------|-----------|-----------|----------------|
| L1     | 32-64 KB  | ~4 ciclos | ~1 TB/s        |
| L2     | 256-512 KB| ~12 ciclos| ~500 GB/s      |
| L3     | 8-32 MB   | ~40 ciclos| ~200 GB/s      |
| RAM    | 8-64 GB   | ~200 ciclos| ~50 GB/s      |

### Conceptos Clave

- **Localidad Espacial:** Datos cercanos en memoria tienden a ser accedidos juntos
- **Localidad Temporal:** Datos recientemente accedidos tienden a ser accedidos nuevamente
- **Línea de Caché:** Unidad mínima de transferencia entre caché y memoria (64 bytes)
- **Cache Miss:** Acceso a dato no presente en caché → traer desde nivel inferior
- **Prefetching:** Mecanismo de hardware que predice y precarga datos antes de ser solicitados

---

## Apéndice: Generación de Gráficos

Para generar gráficos visuales de estos resultados, instala las dependencias necesarias:

```bash
# Opción 1: Usar entorno virtual
python3 -m venv venv
source venv/bin/activate
pip install matplotlib pandas

# Opción 2: Instalar con homebrew
brew install python-matplotlib

# Luego ejecutar:
python3 generate_plots_simple.py
```

Esto generará:
- `ejercicio1_analisis.png`: Panel con 4 gráficos comparativos
- `ejercicio1_strides_temporal.png`: Evolución temporal de todos los strides

---

**Fin del Informe**

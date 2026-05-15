# Resumen Clase 8 - Debugging y Profiling

## 1. Titulo del resumen

**Clase 8 - Debugging y Profiling**

## 2. Temas principales

- Debugging en CUDA como proceso para validar correctitud en codigo host-device y dentro de kernels.
- `cuda-gdb` como herramienta para inspeccionar ejecucion, estado de hilos y memoria en CPU y GPU.
- `compute-sanitizer` como conjunto de analizadores para detectar errores de memoria, sincronizacion e inicializacion.
- Profiling como metodo para observar donde se va el tiempo y por que un programa CUDA no escala o no aprovecha la GPU.
- Diferencia conceptual entre **vision global del sistema** (`Nsight Systems`) y **analisis detallado de kernels** (`Nsight Compute`).
- `NVTX` como mecanismo para anotar regiones del codigo y hacer mas interpretable la ejecucion.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase completa el bloque de programacion CUDA de las clases 5, 6 y 7.
- Las clases anteriores habian mostrado:
  - como estructurar programas CUDA;
  - como mapear hilos y bloques a los datos;
  - como razonar sobre memoria, sincronizacion y rendimiento.
- Esta clase agrega las herramientas para responder dos preguntas clave:
  - **el programa es correcto?**
  - **si es correcto, donde se esta perdiendo rendimiento?**
- Idea global para examen:
  - en CUDA no alcanza con que un kernel compile y produzca algun resultado;
  - hay que poder **validar correctitud** y **medir comportamiento real** sobre CPU, GPU, memoria y sincronizacion.

### Debugging y profiling cumplen roles distintos

- **Debugging** busca encontrar errores de correctitud.
- **Profiling** busca entender comportamiento y rendimiento.
- La diferencia es importante:
  - un programa puede ser lento pero correcto;
  - puede ser rapido en algunos casos pero incorrecto;
  - puede parecer correcto y en realidad fallar solo bajo cierto paralelismo, cierto tamano de entrada o cierto orden de ejecucion.
- En CUDA esta separacion es especialmente importante porque:
  - hay ejecucion asincrona entre host y device;
  - hay miles de hilos concurrentes;
  - hay multiples espacios de memoria;
  - los errores pueden ser no deterministas, como ya sugerian las clases 5 y 6 al hablar de race conditions y sincronizacion.

### Debugging de CUDA: que problemas busca detectar

- El debugging en CUDA apunta a detectar problemas como:
  - kernels que no lanzan o fallan al ejecutarse;
  - accesos invalidos a memoria global, local o shared;
  - variables con valores inesperados en hilos concretos;
  - errores de indice al mapear `grid`, `block` y `thread` a los datos;
  - condiciones de carrera;
  - uso incorrecto de barreras o sincronizacion.
- Conexion con clases anteriores:
  - en clase 5 aparecian race conditions y el problema de coordinar hilos;
  - en clase 6 aparecian errores asincronos y chequeo de errores;
  - en esta clase aparecen herramientas para observar esos problemas de forma sistematica.

### `cuda-gdb`: debugging interactivo de CPU y GPU

- `cuda-gdb` extiende la idea de un debugger clasico al contexto CUDA.
- La presentacion destaca que permite:
  - debugging simultaneo en CPU y GPU;
  - breakpoints;
  - inspeccion de memoria, registros y variables;
  - foco sobre kernels, bloques y threads especificos;
  - trabajo con multiples GPUs, contextos y kernels;
  - debugging a nivel de codigo fuente y ensamblador.
- La idea conceptual importante no es memorizar comandos, sino entender **que tipo de preguntas permite responder**:
  - en que hilo ocurre el problema;
  - si un kernel llego realmente a cierto punto;
  - que valor tiene una variable en memoria global/shared/local;
  - si el error depende de un bloque o thread particular.
- Esto es crucial en CUDA porque muchos errores no se ven mirando solo el flujo del host.

### Como interpretar el valor de `cuda-gdb`

- `cuda-gdb` sirve cuando se sospecha un error **localizable** en la ejecucion:
  - un indice mal calculado;
  - una variable corrupta;
  - una rama inesperada;
  - un thread que escribe donde no debe.
- Es mas util para reconstruir **la historia de ejecucion** de un caso puntual que para medir performance global.
- Compilar con `-g -G` facilita esa inspeccion, pero conceptualmente implica una idea importante:
  - el binario de debugging no representa necesariamente el rendimiento real;
  - las herramientas de correctitud y las de performance no deben confundirse.

### `compute-sanitizer`: validacion automatizada de errores frecuentes

- La presentacion muestra `compute-sanitizer` como una suite de herramientas especializadas.
- Su valor conceptual es alto porque automatiza chequeos que en ejecuciones normales pueden pasar desapercibidos.

### Que detecta cada herramienta

- **Memcheck**:
  - accesos fuera de rango;
  - accesos desalineados.
- **Racecheck**:
  - condiciones de carrera en memoria compartida.
- **Initcheck**:
  - accesos a memoria global no inicializada.
- **Synccheck**:
  - uso incorrecto de sincronizacion.

### Por que estas categorias importan tanto en CUDA

- Cada una ataca una fuente clasica de errores del modelo CUDA:
  - **fuera de rango**: indices mal calculados al mapear threads a datos;
  - **desalineados**: problemas de direccionamiento que pueden afectar correctitud o rendimiento;
  - **race conditions**: varios hilos actualizando o leyendo datos compartidos sin proteccion adecuada;
  - **memoria no inicializada**: kernels que dependen de datos nunca escritos;
  - **sincronizacion incorrecta**: barreras mal ubicadas o supuestos invalidos sobre el orden de ejecucion.
- Esto conecta de forma directa con la teoria previa:
  - la jerarquia de memoria y la concurrencia hacen que el error no siempre sea visible de inmediato;
  - por eso una herramienta que clasifica el tipo de fallo acelera mucho la validacion.

### Relacion entre `compute-sanitizer` y el razonamiento teorico

- La utilidad real no es solo "correr una herramienta", sino usar su salida para formular una hipotesis correcta:
  - si aparece un error de `Racecheck`, hay que revisar cooperacion entre hilos y uso de shared memory;
  - si aparece un error de `Synccheck`, hay que revisar la logica de barreras y quienes alcanzan cada punto;
  - si aparece `Memcheck`, probablemente el problema este en el calculo de indices o tamanos.
- En otras palabras:
  - el sanitizer no reemplaza entender CUDA;
  - sirve porque mapea sintomas observables a conceptos del modelo de ejecucion.

### Profiling: medir antes de optimizar

- La segunda mitad de la clase pasa de correctitud a rendimiento.
- La idea central del profiling es:
  - no optimizar por intuicion;
  - observar que esta haciendo realmente el programa.
- En CUDA esto es especialmente importante porque el tiempo total puede repartirse entre:
  - trabajo del host;
  - lanzamientos de kernels;
  - transferencias de memoria;
  - esperas y sincronizaciones;
  - ejecucion efectiva en GPU.
- Por eso una pregunta como "mi programa CUDA es lento" no alcanza:
  - primero hay que ubicar si el cuello de botella esta en CPU, en GPU o en el movimiento de datos.

### `Nsight Systems`: vision global del sistema

- `Nsight Systems` da una vista integral de la ejecucion.
- La presentacion destaca que permite observar:
  - CPUs;
  - GPUs;
  - hilos de CPU;
  - streams en GPU;
  - kernels;
  - transferencias de memoria.
- Conceptualmente, esta herramienta sirve para responder preguntas de **orquestacion**:
  - la GPU esta ocupada o pasa tiempo ociosa;
  - las transferencias dominan el tiempo total;
  - hay solapamiento entre computo y comunicacion;
  - el host esta alimentando bien a la GPU;
  - cuantos kernels se ejecutan y con que configuracion general.

### Que metricas o observaciones importan en `Nsight Systems`

- Aunque la diapositiva no enumera contadores detallados, deja claro que importan observaciones como:
  - duracion relativa de kernels y copias;
  - secuencia temporal de eventos;
  - concurrencia entre streams;
  - presencia de huecos o tiempos muertos;
  - relacion entre ejecucion del host y del device.
- La interpretacion de rendimiento aca es de nivel macro:
  - si la mayor parte del tiempo se va en copias `host-device`, el problema no esta primero en optimizar instrucciones del kernel;
  - si la GPU tiene pausas entre kernels, puede haber sobrecosto de lanzamiento o mala orquestacion desde CPU;
  - si no hay solapamiento entre transferencia y computo, se esta desaprovechando capacidad del sistema.

### `Nsight Compute`: analisis detallado del kernel

- `Nsight Compute` trabaja a un nivel mas fino.
- La presentacion lo ubica como herramienta para perfilar kernels y extraer metricas.
- Idea conceptual:
  - si `Nsight Systems` dice **donde** mirar,
  - `Nsight Compute` ayuda a entender **por que** un kernel particular rinde mal.
- Por defecto extrae un conjunto chico de metricas, y luego pueden pedirse conjuntos mas detallados.
- Esto refleja una idea metodologica correcta:
  - empezar con una lectura general;
  - profundizar solo cuando ya se identifico el kernel relevante.

### Que tipo de metricas importan al perfilar kernels

- Aunque las diapositivas no listan contadores concretos, por conexion con la clase 6 las metricas importantes son las que permiten interpretar:
  - uso de recursos del kernel;
  - eficiencia del acceso a memoria;
  - grado de ocupacion;
  - costo relativo de instrucciones y esperas.
- En este contexto, para estudiar conviene recordar que el rendimiento de un kernel suele leerse a traves de preguntas como:
  - esta limitado por memoria o por computo;
  - hay suficiente trabajo residente para ocultar latencia;
  - los accesos a memoria son razonables;
  - el uso de recursos por bloque esta restringiendo paralelismo.
- La clase no desarrolla estos contadores, pero si posiciona `Nsight Compute` como la herramienta para obtenerlos.

### Como se interpreta el rendimiento en CUDA

- La interpretacion correcta del rendimiento combina varias capas:
  - **capa sistema**: host, GPU, copias y solapamiento;
  - **capa kernel**: configuracion, recursos y metricas del kernel;
  - **capa conceptual**: relacion con coalescing, ocupacion, sincronizacion y reutilizacion de datos.
- Esto conecta directamente con la clase 6:
  - si un kernel tiene mal rendimiento, la causa puede estar en accesos no coalesced, uso pobre de shared memory, demasiada sincronizacion o mala ocupacion;
  - el profiler permite pasar de esa intuicion teorica a evidencia observable.
- Idea de examen:
  - perfilar no es solo "medir tiempo";
  - es interpretar tiempos y metricas a la luz del modelo CUDA.

### `NVTX`: anotar para entender

- `NVTX` permite marcar regiones del codigo con `nvtxRangePush` y `nvtxRangePop`.
- Su utilidad conceptual es simple pero fuerte:
  - cuando un programa tiene varios kernels, fases y transferencias, las marcas ayudan a relacionar el timeline con la estructura logica del programa.
- En otras palabras:
  - no mejora el rendimiento por si sola;
  - mejora la **observabilidad** del rendimiento.
- Esto vuelve mas claro que parte del programa corresponde a inicializacion, preprocesamiento, kernel principal, reduccion final, etc.

### Metodologia recomendada a partir de la clase

- La clase sugiere implicitamente una metodologia de trabajo:
  1. verificar correctitud basica del programa;
  2. usar herramientas de debugging y sanitizacion para encontrar errores concretos;
  3. perfilar el sistema completo para ubicar el cuello de botella;
  4. perfilar kernels puntuales para entender el origen del costo;
  5. reinterpretar los resultados usando conceptos de memoria, warps, sincronizacion y ocupacion vistos antes.
- Esta secuencia importa porque optimizar antes de validar correctitud suele producir diagnosticos falsos o mejoras irrelevantes.

### Continuidad con los resumenes anteriores

- Conviene reutilizar como contexto:
  - `Clase5-Programacion_CUDA-resumen.md`, porque fija el modelo host-device y el mapeo de threads a datos;
  - `Clase6-Programacion_CUDA_2-resumen.md`, porque explica memoria, coalescing, shared memory, errores asincronos y recomendaciones de performance;
  - `Clase7-Programacion_CUDA_3-resumen.md`, porque refuerza sincronizacion y cooperacion entre hilos, utiles para interpretar race conditions y comportamiento intra-warp.
- Esta clase no reemplaza esas bases:
  - las herramientas sirven justamente para observar problemas que esos modelos teoricos ayudan a explicar.

## 4. Conceptos clave para memorizar

- **Debugging** = buscar errores de correctitud.
- **Profiling** = medir e interpretar rendimiento.
- En CUDA ambas tareas son complementarias pero distintas.
- `cuda-gdb` permite debugging interactivo sobre CPU y GPU.
- `cuda-gdb` sirve para inspeccionar kernels, bloques, threads, memoria y registros.
- Compilar con `-g -G` facilita debugging, pero no representa performance real.
- `compute-sanitizer` agrupa herramientas para detectar errores frecuentes en ejecucion CUDA.
- **Memcheck** detecta accesos fuera de rango y desalineados.
- **Racecheck** detecta condiciones de carrera en memoria compartida.
- **Initcheck** detecta uso de memoria global no inicializada.
- **Synccheck** detecta uso incorrecto de sincronizacion.
- `Nsight Systems` da una vision global del sistema y del timeline de ejecucion.
- `Nsight Compute` analiza kernels en detalle y extrae metricas de rendimiento.
- `NVTX` permite anotar regiones del codigo para hacer mas legible el profiling.
- El rendimiento en CUDA debe interpretarse a varios niveles:
  - host vs device;
  - computo vs transferencias;
  - vision global del sistema vs detalle del kernel.
- Una herramienta no reemplaza el modelo teorico:
  - los resultados se interpretan con conceptos como memoria, ocupacion, sincronizacion y acceso a datos.

## 5. Posibles preguntas teoricas de examen

- Cual es la diferencia conceptual entre debugging y profiling en CUDA?
- Por que en CUDA no alcanza con validar solo el codigo del host?
- Que tipo de errores permite detectar `cuda-gdb`?
- Que diferencia hay entre usar `cuda-gdb` y usar `compute-sanitizer`?
- Que detecta cada subherramienta de `compute-sanitizer` y que problema del modelo CUDA refleja?
- Por que una condicion de carrera puede ser dificil de detectar sin herramientas especializadas?
- Que informacion aporta `Nsight Systems` y para que tipo de diagnostico sirve?
- Que informacion aporta `Nsight Compute` y como complementa a `Nsight Systems`?
- Como se interpreta si el tiempo total del programa esta dominado por transferencias de memoria?
- Que significa usar profiling como herramienta conceptual y no solo como medicion de tiempo?
- Por que `NVTX` ayuda a entender programas CUDA complejos aunque no optimice directamente la ejecucion?
- Como se relacionan las herramientas de profiling con conceptos como coalescing, ocupacion y sincronizacion?

## 6. Dudas o ambigüedades detectadas en la presentación, si las hay

- Hay una inconsistencia de numeracion:
  - el archivo se llama **Clase8-Debugging_y_Profiling.pdf**;
  - dentro de la presentacion aparece **Clase 9: Debugging y Profiling**;
  - y el encabezado general dice **Clase 10 – Ecosistema CUDA (extendido)**.
- La presentacion enumera herramientas y usos, pero desarrolla poco las metricas concretas de profiling.
- `Nsight Compute` se presenta como extractor de metricas, pero no se detallan ejemplos especificos de contadores ni criterios cuantitativos de interpretacion.
- `compute-sanitizer` menciona `Racecheck` sobre memoria compartida; la presentacion no aclara limites exactos del alcance respecto de otros tipos de carrera.

## Nombre de archivo sugerido

- `resumenes/Clase8-Debugging_y_Profiling-resumen.md`

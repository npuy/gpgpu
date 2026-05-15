# Resumen Clase 11 - Ecosistema CUDA

## 1. Titulo del resumen

**Clase 11 - Ecosistema CUDA**

## 2. Temas principales

- Ecosistema CUDA como conjunto de bibliotecas y herramientas que extienden el modelo de programacion CUDA.
- BLAS como antecedente conceptual para entender `CUBLAS` y `NVBLAS`.
- Diferencia entre niveles `BLAS-1`, `BLAS-2` y `BLAS-3`, y por que los niveles mas altos suelen ofrecer mejor potencial de optimizacion.
- Bibliotecas especializadas del ecosistema:
  - `CUBLAS` y `NVBLAS` para algebra lineal densa;
  - `cuSPARSE` para matrices dispersas;
  - `cuSolver` para factorizaciones y problemas clasicos de algebra lineal;
  - `CURAND` para generacion de numeros aleatorios;
  - `NPP` para procesamiento de imagenes y video.
- `CUPTI` como interfaz para construir herramientas de profiling y tracing sobre CUDA.
- `OpenACC` como modelo de paralelizacion por directivas, mas declarativo y cercano al estilo OpenMP.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase cambia de foco respecto a las clases 5, 6, 7 y 8.
- En esas clases el eje habia sido:
  - como escribir kernels;
  - como organizar `grid -> blocks -> threads`;
  - como usar memoria, sincronizacion y herramientas de debugging/profiling.
- Aqui el foco pasa a ser el **ecosistema alrededor de CUDA**:
  - bibliotecas ya optimizadas;
  - interfaces para observabilidad;
  - modelos de programacion de mas alto nivel.
- Idea global importante para examen:
  - programar en GPU no siempre significa escribir todo desde cero con kernels manuales;
  - muchas veces conviene apoyarse en bibliotecas especializadas que encapsulan optimizaciones complejas del hardware.

### Que aporta el ecosistema CUDA

- El ecosistema CUDA debe entenderse como una capa de software sobre el modelo base de CUDA.
- Esa capa ofrece componentes para resolver familias enteras de problemas recurrentes:
  - algebra lineal densa;
  - algebra lineal dispersa;
  - factorizaciones y solvers;
  - generacion de numeros aleatorios;
  - procesamiento de imagenes;
  - profiling y tracing.
- Esto conecta con clases anteriores:
  - en clases 5 a 7 se estudiaba como implementar algoritmos sobre GPU;
  - en esta clase aparece la idea de **reutilizar implementaciones expertas** en lugar de reprogramar cada patron manualmente.
- Lectura conceptual:
  - CUDA no es solo un lenguaje o runtime;
  - es tambien un conjunto de bibliotecas de alto rendimiento orientadas a dominios especificos.

### BLAS como base conceptual para CUBLAS

- Antes de hablar de `CUBLAS`, la presentacion repasa **BLAS** (`Basic Linear Algebra Subprograms`).
- Idea clave:
  - BLAS no es primero una biblioteca concreta;
  - es una **especificacion estandar** para operaciones basicas de algebra lineal.
- Esto es importante porque explica por que distintas plataformas tienen implementaciones diferentes de una misma interfaz conceptual.
- La presentacion remarca dos ideas centrales:
  - las operaciones se agrupan en **tres niveles**;
  - cuanto mayor es el nivel, mayor suele ser el potencial de optimizacion.

### Los tres niveles de BLAS

- **BLAS-1**:
  - operaciones vector-vector y escalares;
  - costo de datos y computo del orden de `O(n)`;
  - incluye copias, intercambios, escalados, producto escalar, normas y sumatorias.
- **BLAS-2**:
  - operaciones matriz-vector;
  - datos y computo del orden de `O(n^2)`;
  - ejemplo representativo: producto matriz-vector.
- **BLAS-3**:
  - operaciones matriz-matriz;
  - datos del orden de `O(n^2)` y computo del orden de `O(n^3)`;
  - incluye multiplicacion de matrices, actualizaciones de rango `k` y resolucion de sistemas triangulares.
- Idea importante para examen:
  - a medida que se sube de nivel, aumenta la **intensidad computacional**;
  - por eso `BLAS-3` suele ofrecer mejores oportunidades de explotar bien la GPU que `BLAS-1`.
- Conexion con la clase 6:
  - en terminos de performance, esto implica mas trabajo por dato transferido y, por tanto, mejor chance de amortizar el costo de memoria.

### Nomenclatura BLAS

- La presentacion recuerda que BLAS mantiene una nomenclatura estandar.
- Esa nomenclatura codifica:
  - tipo de dato;
  - tipo de matriz;
  - operacion.
- Lo importante para estudiar no es memorizar todas las siglas, sino entender que:
  - la interfaz de BLAS esta muy sistematizada;
  - esa estandarizacion facilita portabilidad conceptual entre implementaciones.

### Distintas implementaciones de BLAS

- La clase enumera implementaciones como:
  - `MKL` de Intel;
  - `OpenBLAS`;
  - `ATLAS`;
  - bibliotecas historicas o especificas de otros fabricantes.
- La idea de fondo es que BLAS es un estandar con multiples implementaciones optimizadas segun arquitectura.
- Esto prepara la transicion hacia `CUBLAS`:
  - del mismo modo que en CPU existen implementaciones optimizadas de BLAS,
  - en GPU NVIDIA aparece `CUBLAS` como implementacion orientada a CUDA.

### CUBLAS

- `CUBLAS` es la implementacion de BLAS sobre GPU dentro del ecosistema CUDA.
- Conceptualmente, permite usar operaciones clasicas de algebra lineal densa sin tener que escribir kernels especificos para cada una.
- La presentacion lo ubica historicamente como parte del surgimiento de CUDA y lo asocia al pasaje de primeras implementaciones en simple precision hacia una biblioteca mas completa.
- Idea conceptual fuerte:
  - `CUBLAS` encapsula optimizaciones complejas para operaciones muy estudiadas;
  - por eso suele ser preferible a una implementacion casera salvo que exista una necesidad muy particular.

### Mejoras y evolucion de CUBLAS

- La clase destaca especialmente mejoras sobre `gemm`, la multiplicacion de matrices generales.
- Esto no es casual:
  - `gemm` es una operacion central de `BLAS-3`;
  - y suele ser uno de los mejores ejemplos de computo intensivo donde la GPU puede rendir muy bien.
- Tambien se mencionan estrategias como:
  - `padding`;
  - enfoques hibridos `CPU + GPU`.
- Lectura conceptual:
  - el rendimiento de bibliotecas como `CUBLAS` surge de anos de refinamiento sobre patrones muy conocidos, no solo del hecho de correr en GPU.

### CUBLAS v2 y NVBLAS

- La presentacion marca algunas extensiones de `CUBLAS v2`:
  - uso de un `handler`;
  - escalares accesibles por referencia desde CPU o GPU;
  - funciones `multi-GPU`;
  - funciones `batch`.
- Conceptualmente, esto muestra que la biblioteca evoluciona para adaptarse a escenarios mas complejos:
  - muchas operaciones pequenas;
  - mas de una GPU;
  - mayor flexibilidad de integracion con el programa.
- Tambien aparece `NVBLAS`:
  - apunta a ejecucion dinamica de operaciones `BLAS-3` en mas de una GPU y CPU.
- Idea de examen:
  - dentro del ecosistema CUDA no solo hay kernels individuales;
  - tambien hay capas que distribuyen automaticamente operaciones algebraicas de alto nivel.

### cuSPARSE

- `cuSPARSE` es la biblioteca orientada a algebra lineal **dispersa**.
- La clase menciona:
  - `spmv` y `sptrsv`;
  - soporte para distintos formatos de matriz dispersa;
  - funciones de conversion;
  - tecnicas para precondicionamiento como `IC` e `ILU`;
  - reordenamiento, incluido coloreado de grafos.
- Idea conceptual:
  - los problemas dispersos exigen estructuras y algoritmos distintos de los densos;
  - por eso no alcanza con trasladar directamente las mismas ideas de `CUBLAS`.
- Esta distincion es importante:
  - **denso** y **disperso** no son solo dos tipos de datos;
  - implican distintos patrones de acceso a memoria, distinta localidad y distintas estrategias de optimizacion.

### cuSolver

- `cuSolver` agrupa rutinas para resolver problemas clasicos de algebra lineal, tanto densa como dispersa.
- La presentacion menciona:
  - `LU`;
  - `Cholesky`;
  - `SVD`;
  - `eig`;
  - `QR`.
- Conceptualmente, `cuSolver` debe verse como una capa mas alta que reutiliza y complementa operaciones basicas para resolver tareas matematicas mas ricas.
- Idea de examen:
  - el ecosistema CUDA cubre no solo primitivas de bajo nivel, sino tambien algoritmos numericos completos muy usados en computacion cientifica.

### CUPTI

- `CUPTI` (`CUDA Profiling Tools Interface`) se presenta como una interfaz para **crear herramientas de profiling y tracing** sobre CUDA.
- Esto conecta de manera directa con la clase 8:
  - en esa clase se estudiaban herramientas como `Nsight Systems` y `Nsight Compute` desde el punto de vista del usuario;
  - aqui aparece la interfaz mas baja que permite construir ese tipo de observabilidad.
- La presentacion enumera cuatro APIs:
  - `Activity API`;
  - `Callback API`;
  - `Event API`;
  - `Metric API`.

### Que aporta cada API de CUPTI

- **Activity API**:
  - guarda en forma asincrona trazas de la aplicacion;
  - trabaja con registros de actividad, buffers y colas.
- **Callback API**:
  - permite ejecutar callbacks cuando la aplicacion llama al runtime de CUDA, al driver o cuando ocurren ciertos eventos.
- **Event API**:
  - permite consultar y manejar contadores de eventos del device.
- **Metric API**:
  - consolida metricas de aplicacion a partir de uno o varios eventos.
- Idea conceptual importante:
  - `CUPTI` no es una biblioteca numerica como `CUBLAS`;
  - es una infraestructura de **instrumentacion y medicion**.
- Conexion con la clase 8:
  - mientras `Nsight` responde preguntas de debugging y performance desde herramientas ya hechas,
  - `CUPTI` habilita construir mecanismos propios de medicion y analisis sobre CUDA.

### CURAND

- `CURAND` es la biblioteca para generacion de numeros:
  - **pseudoaleatorios**;
  - **quasialeatorios**.
- La presentacion distingue ambos conceptos:
  - una secuencia pseudoaleatoria es deterministica pero satisface propiedades estadisticas utiles;
  - una secuencia quasialeatoria tambien es deterministica, pero esta diseniada para cubrir un espacio `n`-dimensional de forma mas uniforme.
- Idea conceptual:
  - la generacion de numeros aleatorios en GPU no es un detalle accesorio;
  - es un componente fundamental para simulaciones, Monte Carlo y muchos algoritmos paralelos.

### Host API y device API en CURAND

- La clase indica que `CURAND` incluye dos componentes:
  - una API del **host**;
  - una API para el **device** a traves de archivo cabezal.
- Esto es importante porque conecta con la separacion `host/device` estudiada en la clase 5.
- Segun la modalidad:
  - los numeros pueden generarse en CPU y quedar en memoria del host;
  - o generarse en GPU y quedar en memoria global del device.
- La presentacion tambien destaca un tercer enfoque conceptual:
  - usando funciones directamente en el device, se puede **consumir** los numeros a medida que se generan y evitar escrituras intermedias a memoria.
- Conexion con la clase 6:
  - esto puede ayudar a reducir trafico de memoria cuando el numero aleatorio se usa inmediatamente dentro del kernel.

### Modelo conceptual de funcionamiento de CURAND

- Los numeros son producidos por **generadores**.
- Cada generador encapsula un **estado interno**.
- Se pueden crear varios generadores al mismo tiempo.
- La secuencia producida es **deterministica** dados los mismos parametros iniciales.
- Idea de examen muy importante:
  - "aleatorio" en este contexto no significa no reproducible;
  - por el contrario, la reproducibilidad controlada es una propiedad valiosa para experimentos y debugging.

### Secuencia normal de uso de CURAND

- La secuencia general mostrada por la clase es:
  1. crear el generador;
  2. configurar opciones;
  3. reservar memoria adecuada;
  4. generar numeros;
  5. usarlos;
  6. volver a generar si hace falta;
  7. destruir el generador.
- Para estudiar importa mas el flujo conceptual que los nombres puntuales de funciones:
  - hay un objeto generador con estado;
  - la memoria de salida debe estar en el espacio correcto;
  - la generacion puede integrarse al flujo asincrono de CUDA.

### Asincronia y punteros validos en CURAND

- La presentacion remarca que `curandGenerate()` lanza trabajo y retorna en forma asincrona.
- Esto conecta directamente con clases 5, 6 y 8:
  - en CUDA, lanzar trabajo no implica que el resultado ya este listo;
  - si otro kernel o stream depende de esos numeros, hay que sincronizar correctamente.
- Tambien subraya una regla basica pero examinable:
  - no se puede pasar un puntero de host cuando se espera memoria del device, ni viceversa.
- En otras palabras:
  - `CURAND` respeta completamente la frontera `host/device` del modelo CUDA.

### NPP

- `NPP` (`NVIDIA Performance Primitives`) se presenta como una biblioteca de funciones para acelerar aplicaciones con GPU de manera sencilla.
- La diapositiva indica que:
  - se centra en procesamiento de imagenes y video;
  - incluye mas de 5000 primitivas.
- Idea conceptual:
  - al igual que `CUBLAS` en algebra lineal, `NPP` busca ofrecer primitivas ya optimizadas para un dominio concreto.
- La presentacion no desarrolla ejemplos ni clasificacion interna, por lo que solo puede afirmarse con seguridad que:
  - es una biblioteca extensa;
  - orientada a tareas frecuentes de multimedia.

### OpenACC

- `OpenACC` aparece al final como enfoque de **paralelismo sencillo**.
- La clase lo describe como:
  - paralelizacion mediante **clausulas/directivas**;
  - una extension de estilo `OpenMP`;
  - con inclusion de las transferencias.
- Conceptualmente, esto lo diferencia del CUDA mas explicito estudiado en clases 5 a 7:
  - en CUDA el programador controla de manera directa kernels, memoria y lanzamientos;
  - en `OpenACC` el programador declara regiones o bucles paralelizables y delega mas decisiones al compilador/runtime.
- Idea de examen:
  - `OpenACC` representa un modelo de mas alto nivel, con menos control fino pero tambien con menor complejidad de programacion.
- La presentacion no entra en detalles semanticos ni en comparaciones profundas con CUDA, asi que no conviene extrapolar mas alla de eso.

### Lectura integradora de la clase

- La clase muestra que el trabajo sobre GPU puede pensarse en varios niveles:
  - **nivel bajo**: escribir kernels CUDA manualmente;
  - **nivel intermedio**: reutilizar bibliotecas especializadas como `CUBLAS`, `cuSPARSE`, `cuSolver`, `CURAND` o `NPP`;
  - **nivel de herramientas**: observar y medir con `CUPTI`;
  - **nivel mas declarativo**: paralelizar con directivas usando `OpenACC`.
- Esa lectura integra bien el recorrido del curso:
  - primero se aprendio el modelo base;
  - luego como optimizarlo y depurarlo;
  - ahora se ubican componentes del ecosistema que permiten escalar productividad y rendimiento sin reimplementar todo.

## 4. Conceptos clave para memorizar

- `BLAS` es una **especificacion**, no una unica biblioteca.
- `BLAS-1`, `BLAS-2` y `BLAS-3` distinguen operaciones vector-vector, matriz-vector y matriz-matriz.
- Operaciones de nivel mas alto, especialmente `BLAS-3`, suelen tener mejor potencial de optimizacion en GPU.
- `CUBLAS` implementa BLAS sobre GPU para algebra lineal densa.
- `NVBLAS` apunta a ejecucion dinamica de operaciones `BLAS-3` en multiples GPUs y CPU.
- `cuSPARSE` atiende algebra lineal dispersa.
- `cuSolver` ofrece algoritmos numericos como `LU`, `Cholesky`, `SVD`, `eig` y `QR`.
- `CUPTI` sirve para tracing, callbacks, eventos y metricas; no para calculo numerico.
- `CURAND` genera secuencias pseudoaleatorias y quasialeatorias.
- En `CURAND`, las secuencias son deterministicas dado el mismo estado inicial.
- `NPP` se orienta a procesamiento de imagenes y video.
- `OpenACC` usa directivas y se parece conceptualmente a `OpenMP`.
- En todo el ecosistema sigue vigente la separacion `host/device` y la asincronia propias de CUDA.

## 5. Posibles preguntas teoricas de examen

- Que se entiende por ecosistema CUDA y por que no se reduce solo a escribir kernels?
- Por que BLAS se describe como una especificacion y no como una biblioteca unica?
- Cual es la diferencia conceptual entre `BLAS-1`, `BLAS-2` y `BLAS-3`?
- Por que las operaciones `BLAS-3` suelen ser mejores candidatas para alto rendimiento en GPU?
- Que problema resuelve `CUBLAS` y cuando tendria sentido preferirlo frente a escribir un kernel propio?
- Que diferencia conceptual hay entre `CUBLAS` y `cuSPARSE`?
- Que tipo de algoritmos agrupa `cuSolver`?
- Para que sirve `CUPTI` y como se relaciona con las herramientas de profiling vistas antes?
- Cual es la diferencia entre la `Activity API`, `Callback API`, `Event API` y `Metric API` de `CUPTI`?
- Que diferencia hay entre secuencias pseudoaleatorias y quasialeatorias en `CURAND`?
- Por que es importante que `CURAND` sea deterministico bajo las mismas condiciones iniciales?
- Que implicancias tiene que `curandGenerate()` sea asincrono?
- Que rol cumple `NPP` dentro del ecosistema CUDA?
- En que se diferencia conceptualmente `OpenACC` del estilo de programacion CUDA explicito?

## 6. Dudas o ambigüedades detectadas en la presentación

- La parte de `NPP` es muy breve: solo indica dominio de aplicacion y cantidad de primitivas, sin clasificacion ni ejemplos.
- La parte de `OpenACC` tambien es muy sintetica: solo lo ubica como paralelizacion por clausulas al estilo OpenMP e indica que incluye transferencias.
- En `CUBLAS`, la presentacion menciona mejoras historicas y extensiones como `batch` o `multi-GPU`, pero no explica criterios de uso ni limitaciones.
- En `CUPTI`, se enumeran las APIs y su objetivo general, pero no se muestran casos de uso concretos ni la relacion exacta con herramientas como `Nsight`.

## Resumenes previos que conviene reutilizar como contexto

- `resumenes/Clase5-Programacion_CUDA-resumen.md`: para mantener la separacion conceptual `host/device`, kernels y asincronia.
- `resumenes/Clase6-Programacion_CUDA_2-resumen.md`: para conectar intensidad computacional, costo de memoria y optimizacion.
- `resumenes/Clase7-Programacion_CUDA_3-resumen.md`: para sostener terminologia de cooperacion y extensiones del modelo CUDA.
- `resumenes/Clase8-Debugging_y_Profiling-resumen.md`: para enlazar `CUPTI` con profiling, tracing y observabilidad.

## Nombre de archivo sugerido

- `resumenes/Clase11-Ecosistema_CUDA-resumen.md`

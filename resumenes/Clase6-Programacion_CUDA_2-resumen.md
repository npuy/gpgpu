# Resumen Clase 6 - Programacion CUDA II

## 1. Titulo del resumen

**Clase 6 - Programacion CUDA II**

## 2. Temas principales

- Coalescing en memoria global y su impacto directo en el uso del ancho de banda.
- Memoria compartida como espacio de cooperacion y como cache administrada por el programador.
- Conflictos de bancos en shared memory.
- Tiling como patron para reutilizar datos y reordenar accesos.
- Manejo correcto de errores en CUDA, distinguiendo errores inmediatos y asincronos.
- Codigo PTX como herramienta para inspeccionar decisiones del compilador.
- Recomendaciones practicas de performance vinculadas con warps, bloques y ocupacion.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase debe leerse como continuacion directa de **Programacion CUDA**.
- La clase 5 habia fijado el modelo basico:
  - host y device;
  - kernels;
  - jerarquia `grid -> blocks -> threads`;
  - cooperacion dentro del bloque;
  - uso general de la memoria global y shared.
- Esta clase avanza sobre una pregunta mas exigente:
  - no solo **como escribir** un programa CUDA correcto,
  - sino **como organizar memoria, sincronizacion y diagnostico** para que el programa use bien la GPU.
- Idea global:
  - en CUDA el rendimiento no depende solo de lanzar muchos hilos;
  - depende de que los warps accedan bien a memoria global, reutilicen datos en shared memory y eviten patrones que serialicen accesos.

### Acceso coalesced a memoria global

- La presentacion enfatiza que el acceso a **memoria global** ocurre por **segmentos** y **transacciones**, no palabra por palabra de forma idealizada.
- Aunque un hilo pida solo una palabra, el hardware transfiere un bloque de memoria alineado.
- Consecuencia:
  - si los hilos de un warp usan bien los datos traidos en una transaccion, el ancho de banda se aprovecha;
  - si cada hilo cae en segmentos distintos o mal alineados, se generan mas transacciones y mucho trafico inutil.
- La idea de **coalescing** es justamente fusionar los accesos de los hilos de un warp en la menor cantidad posible de transacciones.
- Caso favorable:
  - hilos consecutivos acceden a direcciones consecutivas;
  - el warp aprovecha practicamente todo lo transferido.
- Caso desfavorable:
  - hilos consecutivos acceden con stride grande o disperso;
  - cada hilo puede terminar forzando una transaccion separada;
  - gran parte del trafico se desperdicia.
- Relacion con la clase 5:
  - antes importaba mapear hilos a datos;
  - ahora importa **como ese mapeo se refleja en direcciones de memoria dentro del warp**.
- Idea de examen:
  - un kernel puede ser correcto y paralelizar bien el trabajo, pero aun asi rendir mal si el patron de acceso global no es coalesced.

### Alineacion y organizacion de datos

- La clase remarca que el costo depende no solo de que los accesos sean consecutivos, sino tambien de la **alineacion** respecto de los segmentos de memoria.
- Un acceso no alineado obliga a mas transacciones y reduce eficiencia.
- Aparece una decision de diseño de datos muy importante:
  - **AoS** (`Array of Structs`);
  - **SoA** (`Struct of Arrays`).
- La comparacion ilustra que:
  - si muchos hilos quieren el mismo campo de elementos consecutivos,
  - **SoA** suele favorecer accesos contiguos y coalesced;
  - **AoS** introduce separacion entre campos y empeora la localidad para ese patron.
- Esta es una extension importante del modelo CUDA:
  - no solo se decide cuantos hilos lanzar;
  - tambien se decide **como organizar los datos en memoria** para que esos hilos lean y escriban eficientemente.

### Escrituras concurrentes y race conditions en memoria global

- La presentacion recuerda un caso particular importante:
  - si varios hilos de un warp escriben en la misma direccion de memoria global con una instruccion no atomica,
  - aparece una **race condition**.
- CUDA no garantiza cual de esos hilos termina realizando efectivamente la escritura.
- Esto conecta con la clase 5:
  - ya se habia introducido el problema de actualizaciones concurrentes;
  - aqui se subraya que en memoria global el comportamiento puede quedar directamente indefinido.

### Shared memory: extension practica del bloque como unidad de cooperacion

- En la clase 5, el bloque ya aparecia como unidad de cooperacion.
- En esta clase, la **memoria compartida** pasa a ocupar un papel mas estrategico:
  - es mucho mas rapida que la memoria global;
  - existe una por multiprocesador;
  - su alcance sigue siendo a nivel de bloque;
  - su vida termina cuando termina el bloque.
- Conceptualmente, shared memory puede entenderse de dos maneras complementarias:
  - como espacio para compartir datos entre hilos del bloque;
  - como una **cache programada manualmente** por el desarrollador.
- Esto es una diferencia fuerte frente a CPU:
  - en CPU suele confiarse mas en caches automaticas;
  - en CUDA muchas optimizaciones importantes requieren decidir explicitamente que datos copiar a shared, cuando sincronizar y cuando reutilizarlos.

### Formas de reserva de shared memory

- La clase muestra dos modalidades:
  - reserva **estatica**, con tamaño conocido al compilar;
  - reserva **dinamica**, usando `extern __shared__`.
- La idea teorica importante no es la sintaxis, sino la decision de uso:
  - si el tamaño del buffer compartido depende del problema o del lanzamiento, conviene memoria dinamica;
  - si la estructura es fija y clara, la reserva estatica simplifica el diseño.
- Tambien aparece la posibilidad de partir una misma region dinamica en varios arreglos logicos.
- Esto muestra que shared memory es un recurso explicito, escaso y administrado a mano.

### Bancos y bank conflicts

- La shared memory no se comporta como un espacio uniforme ideal:
  - esta dividida en **bancos**.
- Palabras contiguas de 32 bits caen en bancos contiguos.
- Un warp puede acceder en paralelo a shared memory **solo si los accesos se distribuyen bien entre bancos**.
- Si varios hilos acceden a bancos distintos:
  - el acceso puede resolverse simultaneamente.
- Si varios hilos acceden a palabras distintas del mismo banco:
  - aparece un **bank conflict**;
  - el acceso se serializa parcial o totalmente.
- Hay un caso favorable especial:
  - si varios hilos leen exactamente la misma palabra, puede usarse **broadcast** y evitar conflicto.
- Idea conceptual clave:
  - shared memory es rapida, pero **no gratis**;
  - para que realmente acelere, tambien hay que cuidar el patron de acceso dentro del warp.
- En otras palabras:
  - pasar datos de global a shared puede mejorar mucho;
  - pero una mala disposicion en shared puede reintroducir serializacion por otra via.

### Shared memory para reordenar accesos

- La clase no presenta shared memory solo como memoria temporal rapida.
- Tambien la propone como herramienta para **transformar patrones de acceso**.
- Idea central:
  - cargar datos desde memoria global a shared;
  - reorganizarlos o reutilizarlos localmente;
  - luego continuar el calculo reduciendo accesos globales costosos o evitando accesos no coalesced.
- Esta es una de las abstracciones mas importantes de la clase:
  - shared memory permite desacoplar parcialmente el patron de acceso global del patron de computo local.

### Tiling

- El patron que sistematiza esa idea es **tiling**.
- Consiste en dividir los datos globales en **tiles** o porciones manejables y trabajar sobre una o pocas porciones a la vez.
- Esquema general:
  - identificar un conjunto de datos que varios hilos reutilizaran;
  - cargarlo desde memoria global a shared memory;
  - sincronizar para asegurar que la carga termino;
  - computar usando datos en shared;
  - volver a sincronizar si hace falta;
  - avanzar al siguiente tile.
- La razon de fondo es doble:
  - **reutilizacion**: traer un dato una vez y usarlo varias veces;
  - **localidad/control**: convertir accesos globales costosos o dispersos en accesos locales mas convenientes.
- Conexion con contenidos anteriores:
  - en clase 5 se habia mostrado que no alcanza con repartir trabajo;
  - aqui se formaliza una tecnica para que ese reparto se alinee mejor con la jerarquia de memoria.
- Idea de examen:
  - tiling no es solo una optimizacion puntual;
  - es un patron general para acercar datos al bloque y amortizar el costo de memoria global.

### Errores en tiempo de ejecucion: CUDA no falla donde uno espera

- La clase introduce un punto practico muy importante:
  - en CUDA, muchos errores no se reportan inmediatamente donde se originan.
- La presentacion dice explicitamente que los errores pueden ser **asincronos**.
- Esto obliga a distinguir dos clases de chequeo:
  - errores devueltos por llamadas de la API de CUDA;
  - errores asociados al lanzamiento o a la ejecucion real del kernel.
- Idea central:
  - si no se chequean explicitamente los estados de error, un programa puede parecer correcto o fallar en una linea posterior que no es la causa real.

### Chequeo de errores de API

- Las operaciones como reservar memoria, copiar o liberar ya devuelven un `cudaError_t`.
- Por eso conviene envolverlas en un wrapper de chequeo como `CUDA_CHK`.
- Conceptualmente, esto no es un detalle de estilo:
  - es parte de un metodo correcto de trabajo en CUDA.
- La clase subraya que no alcanza con revisar el error al final del programa:
  - hay que revisar **cada llamada relevante**.

### Chequeo de errores de kernel

- En kernels, el esquema cambia porque el lanzamiento es asincrono.
- La clase distingue dos momentos distintos:
  - error de **configuracion o lanzamiento** del kernel;
  - error durante la **ejecucion** del kernel.
- Para capturarlos correctamente, recomienda:
  - llamar a `cudaGetLastError()` inmediatamente despues del kernel;
  - llamar luego a `cudaDeviceSynchronize()`.
- Interpretacion:
  - `cudaGetLastError()` detecta problemas como una configuracion invalida;
  - `cudaDeviceSynchronize()` fuerza a esperar el fin de ejecucion y permite descubrir fallos ocurridos dentro del kernel, como accesos ilegales a memoria.
- Esta distincion es importante para examen porque muestra que en CUDA:
  - lanzar un kernel no implica que ya haya corrido;
  - y un error de lanzamiento no es lo mismo que un error interno de ejecucion.

### PTX como nivel intermedio para inspeccionar performance

- La clase introduce **PTX** como codigo intermedio generado por `nvcc`.
- No se presenta para programar directamente en PTX, sino para **inspeccionar** que decisiones tomo el compilador.
- Idea central:
  - el codigo fuente CUDA puede inducir operaciones distintas a las que el programador imagina;
  - mirar PTX permite detectar conversiones, tipos y operaciones que impactan en performance.
- El ejemplo de la clase es muy ilustrativo:
  - usar `2.27` lleva a conversiones y multiplicacion en doble precision;
  - usar `2.27f` mantiene la operacion en simple precision.
- Leccion conceptual:
  - pequenos detalles del codigo fuente pueden cambiar el tipo de instrucciones generadas;
  - en GPU eso puede afectar costos de computo y throughput.
- Por eso PTX funciona como herramienta diagnostica:
  - no reemplaza el modelo de programacion,
  - pero ayuda a verificar si el compilador materializo lo que el programador pretendia.

### Recomendaciones adicionales de performance

- La clase cierra reforzando ideas que ahora tienen fundamento mas concreto:
  - evitar divergencia dentro del warp;
  - usar una cantidad de bloques mayor que la de multiprocesadores para mantener ocupacion;
  - preferir cantidad de hilos por bloque multiplo de 32, porque el warp tiene 32 hilos.
- En particular, sugiere incluso usar mas de dos bloques por multiprocesador potencial para ayudar a ocultar latencias, por ejemplo cuando un bloque queda esperando en una barrera.
- Esto enlaza de manera directa con la clase 4:
  - la GPU sostiene rendimiento alternando warps y bloques listos;
  - si no hay suficiente trabajo residente, esa capacidad de ocultar latencia se desaprovecha.

### Que agrega esta clase al modelo CUDA

- Si la clase 5 explicaba **como estructurar** un programa CUDA, la clase 6 explica **como decidir mejor**:
  - como ordenar datos en memoria;
  - cuando usar shared memory;
  - como pensar shared como cache manual;
  - como evitar desperdicio de ancho de banda;
  - como detectar si el fallo esta en la API, en el lanzamiento o en la ejecucion;
  - como usar PTX para observar efectos no obvios del codigo fuente.
- En resumen:
  - el modelo CUDA no termina en `threads`, `blocks` y `kernels`;
  - empieza a volverse realmente util cuando se conecta con **patrones de acceso, reutilizacion de datos y diagnostico correcto de errores**.

## 4. Conceptos clave para memorizar

- **Coalesced access** = accesos de un warp que pueden fusionarse en pocas transacciones de memoria global.
- El rendimiento en memoria global depende de:
  - contiguidad;
  - alineacion;
  - distribucion de direcciones dentro del warp.
- Un acceso global no coalesced desperdicia ancho de banda aunque el kernel sea correcto.
- **SoA** suele favorecer accesos coalesced cuando muchos hilos leen el mismo campo de elementos consecutivos.
- **Shared memory**:
  - es rapida;
  - es por bloque;
  - se gestiona explicitamente;
  - sirve tanto para cooperacion como para reutilizacion/cache manual.
- **Bank conflict** = varios hilos del warp acceden a palabras distintas del mismo banco de shared memory y el acceso se serializa.
- **Broadcast** = varios hilos leen la misma palabra de shared memory sin conflicto.
- **Tiling** = procesar datos por porciones cargadas a shared para reutilizarlos y reducir o mejorar accesos a memoria global.
- En CUDA los errores pueden ser **asincronos**.
- Para API CUDA: conviene envolver cada llamada con chequeo de error.
- Para kernels:
  - `cudaGetLastError()` detecta errores de lanzamiento/configuracion;
  - `cudaDeviceSynchronize()` detecta errores ocurridos durante la ejecucion.
- **PTX** = representacion intermedia que permite inspeccionar decisiones del compilador con impacto en performance.
- Elegir mal tipos numericos o literales puede introducir conversiones e instrucciones mas costosas.
- Recomendaciones base:
  - evitar divergencia dentro del warp;
  - usar mas bloques que multiprocesadores;
  - elegir hilos por bloque multiplos de 32.

## 5. Posibles preguntas teoricas de examen

- Que significa que un acceso a memoria global sea coalesced y por que eso afecta tanto la performance?
- Por que dos kernels igualmente correctos pueden tener rendimientos muy distintos segun el patron de acceso a memoria?
- En que situaciones una organizacion de datos **SoA** puede ser preferible a **AoS** en GPU?
- Cual es el doble rol de la shared memory dentro de CUDA?
- Que es un **bank conflict** y por que shared memory puede perder rendimiento a pesar de ser muy rapida?
- Que es **tiling** y que problema busca resolver?
- Por que en CUDA no alcanza con chequear errores solo al final del programa?
- Que diferencia hay entre un error de lanzamiento de kernel y un error durante la ejecucion del kernel?
- Para que sirve inspeccionar codigo PTX si el programador escribe en CUDA C/C++?
- Por que conviene que el numero de hilos por bloque sea multiplo de 32?

## 6. Dudas o ambiguedades detectadas en la presentacion

- La presentacion mezcla distintas granularidades al hablar de segmentos de 128 bytes y transacciones de 32 bytes. La idea general de coalescing queda clara, pero el detalle exacto depende de la arquitectura y de la compute capability.
- En la seccion de tiling aparece la aclaracion "tiles (bloques?)", lo que sugiere una simplificacion terminologica. Conviene distinguir:
  - **tile** como porcion de datos/procesamiento;
  - **block** como unidad de ejecucion de CUDA.
- La presentacion menciona PTX como herramienta de optimizacion, pero no profundiza en hasta donde conviene analizarlo ni como cambia entre arquitecturas.

## Resumenes previos que conviene reutilizar

- [Clase5-Programacion_CUDA-resumen.md](/Users/nicolaspereira/Documents/facu/gpgpu/resumenes/Clase5-Programacion_CUDA-resumen.md) como contexto principal, porque esta clase lo extiende directamente.
- [Clase4-Arquitectura_de_la_GPU-resumen.md](/Users/nicolaspereira/Documents/facu/gpgpu/resumenes/Clase4-Arquitectura_de_la_GPU-resumen.md) para repasar warps, jerarquia de memoria y ocultamiento de latencia, que explican por que estas optimizaciones importan.

## Nombre de archivo propuesto

- `resumenes/Clase6-Programacion_CUDA_2-resumen.md`

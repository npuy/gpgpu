# Resumen Clase 4 - Arquitectura de la GPU

## 1. Titulo del resumen

**Clase 4 - Arquitectura de la GPU**

## 2. Temas principales

- Arquitectura CUDA inicial basada en G80.
- Organizacion jerarquica del hardware: GPU, clusters, multiprocesadores y CUDA cores.
- Relacion entre throughput, ocultamiento de latencia y paralelismo masivo.
- Modelo de ejecucion clasico de CUDA: SPMD, SIMT, blocks, grids y warps.
- Jerarquia de memoria clasica y alcance de cada espacio de memoria.
- Evolucion arquitectonica desde Fermi hasta Turing.
- Compute capabilities como forma estandar de describir generaciones y capacidades.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- La clase explica como esta organizada internamente una GPU CUDA y por que ese hardware obliga a pensar la programacion de cierta manera.
- Conexion con las clases anteriores:
  - La clase 1 habia mostrado que la GPU se diseña para **throughput** mas que para **latency**.
  - La clase 2 habia mostrado que el rendimiento moderno depende de explotar **paralelismo de datos**.
  - La clase 3 habia mostrado los problemas practicos de repartir trabajo, sincronizar y evitar contencion.
- Esta clase une todo eso en una idea concreta:
  - la GPU logra rendimiento porque organiza muchisimos hilos sobre una jerarquia de multiprocesadores, con poco enfasis en caches tradicionales y mucho enfasis en mantener siempre trabajo listo para ejecutar.
- En otras palabras:
  - CUDA no se programa con muchos threads por comodidad;
  - se programa con muchos threads porque el hardware fue construido para esconder latencia y sostener ocupadas las unidades de ejecucion.

### Arquitectura CUDA inicial: G80

- La arquitectura CUDA puede verse como un **arreglo escalable de multiprocesadores**.
- En la arquitectura **G80**, la primera orientada a CUDA:
  - hay **16 SMs** (`Streaming Multiprocessors`);
  - los SMs se agrupan de a pares en **8 TPCs** (`Thread Processing Clusters`);
  - los TPCs contienen caches de texturas y de memoria constante compartidos por sus SMs.
- Cada **SM** del G80 contiene:
  - **8 procesadores escalares** o **Streaming Processors (SP)**, luego conocidos como **CUDA cores**;
  - una unidad de instrucciones multihilo;
  - memoria compartida on-chip;
  - **2 SFUs** (`Special Function Units`) para operaciones trascendentales como `sin`, `cos`, `log` y `sqrt`.
- Idea conceptual importante:
  - el CUDA core no se presenta como un nucleo autonomo al estilo CPU;
  - es una unidad de calculo simple dentro de una estructura mayor que comparte registros, planificacion y otros recursos a nivel de SM.
- La presentacion remarca que los CUDA cores:
  - son basicamente ALUs;
  - no tienen caches de datos propias;
  - no tienen registros propios independientes del multiprocesador.
- Esto ayuda a entender por que la GPU puede integrar muchos mas recursos de calculo que una CPU:
  - simplifica el core individual;
  - concentra el control y la administracion de contexto a nivel del SM;
  - dedica mas area del chip al computo efectivo.

### Por que la GPU necesita paralelismo masivo

- Una CPU tradicional usa caches complejas y mucho hardware de control para **reducir la latencia** de cada hilo.
- La GPU sigue otra estrategia:
  - reduce o elimina gran parte de ese cache de datos clasico;
  - cuando un conjunto de hilos espera datos de memoria, se suspende;
  - inmediatamente otro conjunto de hilos pasa a ejecutar;
  - esa planificacion se hace en hardware, en forma transparente y con costo muy bajo.
- Esta es una de las ideas mas importantes de toda la clase:
  - la GPU **no oculta latencia haciendo cada hilo mas inteligente**;
  - la GPU **oculta latencia teniendo muchisimos hilos listos para alternar**.
- Consecuencia directa:
  - hacen falta **muchos mas hilos que unidades de ejecucion**.
- En G80, cada SM puede tener hasta **768 hilos concurrentes** residentes, aunque no ejecutan todos literalmente al mismo tiempo.
- Con 16 SMs, la GPU podia mantener **12288 hilos concurrentes**.
- Conexion con CUDA:
  - esto explica por que el programador no lanza unos pocos threads "pesados", como en CPU;
  - en GPU conviene expresar el problema como miles de tareas pequenas, similares e independientes.
- Relacion con clases 2 y 3:
  - la **descomposicion de dominio** es la natural para este hardware;
  - el **balance de carga** se vuelve mas simple cuando el trabajo se divide en muchas unidades pequenas;
  - la **sincronizacion costosa** debe minimizarse porque rompe la capacidad de ocultar latencia con alternancia de hilos.

### Modelo de ejecucion clasico: de SPMD a SIMT

- La GPU se abstrae como un conjunto de multiprocesadores, y cada multiprocesador ejecuta el mismo programa sobre datos distintos.
- A nivel global, el paradigma se describe como **SPMD** (`Single Program Multiple Data`):
  - muchos hilos ejecutan el mismo kernel;
  - distintos multiprocesadores no tienen por que estar en la misma instruccion al mismo tiempo.
- A nivel interno de cada multiprocesador, el paradigma es **SIMT** (`Single Instruction Multiple Threads`).
- La relacion entre ambos niveles es importante:
  - **SPMD** describe el modelo de programacion visto por el programador;
  - **SIMT** describe como el hardware organiza realmente la ejecucion de grupos de hilos.
- La clase compara **SIMT** con **SIMD**:
  - ambos difunden una instruccion sobre multiples elementos de procesamiento;
  - pero en **SIMT** cada carril puede verse como un hilo separado, con identidad propia;
  - el hardware mantiene informacion sobre que hilos siguen activos o han divergido.
- Distincion conceptual importante:
  - en **SIMD**, la vision dominante es la de una instruccion vectorial sobre varios datos;
  - en **SIMT**, la vision dominante es la de muchos hilos que el hardware agrupa y secuencia de forma conjunta cuando es posible.
- Esto conecta directamente con la clase 3:
  - la divergencia que antes aparecia como un problema general de control de flujo ahora adquiere un costo arquitectonico concreto.

### Kernels, grids, blocks y escalabilidad

- Los programas que corren en la GPU se llaman **kernels**.
- Segun esta presentacion, la GPU ejecuta un kernel a la vez.
- La ejecucion de un kernel se organiza asi:
  - los **hilos** se agrupan en **bloques**;
  - los **bloques** se agrupan en un **grid**.
- Los bloques pueden ser:
  - unidimensionales;
  - bidimensionales;
  - tridimensionales.
- El grid puede ser:
  - unidimensional;
  - bidimensional.
- Cada bloque representa un subconjunto del trabajo total que puede ejecutarse en un multiprocesador en forma independiente.
- Si hay suficientes MPs disponibles, varios bloques del grid se ejecutan en paralelo.
- Si no los hay:
  - el scheduler decide cuando corre cada bloque;
  - un mismo MP puede ejecutar multiples bloques;
  - el orden real de ejecucion de bloques no se conoce a priori.
- Idea conceptual clave para examen:
  - la organizacion `grid -> blocks -> threads` permite **escalabilidad automatica**.
- Esto explica varias decisiones del modelo CUDA:
  - los bloques deben ser relativamente independientes;
  - no se debe asumir orden entre bloques;
  - el grid expresa mucho paralelismo potencial para que el hardware lo mapee segun los recursos disponibles.
- Dicho de otra forma:
  - CUDA separa la descripcion del paralelismo de su asignacion fisica exacta al hardware.
  - El programador define el trabajo potencial; el hardware decide como lo distribuye.

### Warps: la unidad real de planificacion

- Aunque el programador piensa en threads y blocks, el hardware planifica en base a **warps**.
- Un **warp** es un grupo de hilos consecutivos que el multiprocesador crea al particionar un bloque.
- En G80, el tamano del warp es **32 hilos**.
- El warp es la unidad sobre la que el SM:
  - crea contexto de ejecucion;
  - planifica;
  - despacha instrucciones.
- La presentacion dice explicitamente que un warp ejecuta **una instruccion por vez**.
- Si los hilos de un warp encuentran una bifurcacion dependiente de datos:
  - el warp serializa los caminos;
  - deshabilita temporalmente los hilos que no participan en cada camino;
  - cuando terminan los caminos, los hilos convergen.
- Consecuencia central:
  - la maxima eficiencia aparece cuando los hilos de un mismo warp siguen el mismo camino de ejecucion.
- La divergencia ocurre **dentro del warp**.
- Distintos warps se ejecutan en forma independiente.
- Esta es probablemente la razon mas fuerte de por que CUDA se programa como se programa:
  - conviene que threads cercanos trabajen sobre datos similares y con control similar;
  - conviene evitar ramas donde hilos del mismo warp tomen decisiones distintas;
  - conviene elegir tamanos y organizaciones de bloque que produzcan buen aprovechamiento de warps completos.
- Conexion con la clase 3:
  - antes la divergencia era una dificultad conceptual;
  - ahora queda claro que en GPU la divergencia **reduce paralelismo efectivo** al serializar partes del trabajo.

### Jerarquia de memoria clasica

- La clase distingue varios espacios de memoria en el dispositivo:
  - **global**;
  - **local**;
  - **compartida**;
  - **registros**;
  - **constante**;
  - **texturas**.
- Tambien distingue entre memorias:
  - **on-chip**: en el mismo chip que los CUDA cores;
  - **off-chip**: fuera de ese chip.
- La jerarquia importa porque en GPU el rendimiento depende enormemente de donde viven los datos y quien puede accederlos.

### Memoria global

- La **memoria global**:
  - es visible por todos los hilos del dispositivo;
  - es **off-chip**;
  - en el modelo clasico presentado es la memoria mas lenta y no cacheada.
- Conceptualmente, es el gran espacio comun del device.
- Implicancia para CUDA:
  - es el lugar natural para datos grandes;
  - pero no conviene tratarla como si tuviera el costo de una variable comun de CPU.

### Registros

- Los **registros**:
  - son **on-chip**;
  - son la memoria mas rapida de la GPU;
  - son accesibles solo por cada hilo;
  - son gestionados por el compilador.
- Conexion con la clase 3:
  - representan el caso mas claro de **estado privado por hilo**.
- Implicancia:
  - el trabajo de cada thread debe aprovechar al maximo su estado local en registros antes de recurrir a memorias mas lentas.

### Memoria local

- La **memoria local**:
  - es privada de cada hilo;
  - es **off-chip**;
  - es una de las memorias mas lentas;
  - es gestionada por el compilador.
- La presentacion aclara una posible confusion:
  - aunque el diagrama la muestre "cerca" del hilo por ser privada, fisicamente no es on-chip.
- Punto conceptual importante:
  - "local" no significa "rapida";
  - significa "de alcance por hilo".
- Esto suele ser importante para examen y para entender CUDA:
  - el nombre describe visibilidad logica, no costo.

### Memoria compartida

- La **memoria compartida**:
  - pertenece a cada bloque;
  - puede ser accedida por cualquier hilo del bloque;
  - es **on-chip**;
  - es casi tan rapida como los registros;
  - vive mientras vive el bloque.
- Esta memoria explica por que CUDA organiza hilos en bloques y no solo en un conjunto plano:
  - el bloque no es solo una conveniencia de indices;
  - es tambien la **unidad de cooperacion y de memoria compartida rapida**.
- Conexion fuerte con la clase 3:
  - la clase anterior habia mostrado la importancia de variables compartidas, sincronizacion y reducciones;
  - la shared memory es el lugar natural para cooperacion eficiente dentro de un bloque.
- Implicancia:
  - muchas optimizaciones en CUDA consisten en traer datos desde memoria global a memoria compartida, reutilizarlos varias veces y reducir accesos lentos.

### Memoria constante y texturas

- La **memoria constante**:
  - es solo de lectura para el dispositivo;
  - es **off-chip**, pero cacheada;
  - puede verse casi como un cache especializado para ciertos accesos.
- Las **texturas** tienen caracteristicas similares en esta presentacion.
- Idea conceptual:
  - no toda memoria off-chip tiene el mismo comportamiento;
  - hay espacios especializados segun patron de acceso y semantica del dato.

### Resumen conceptual de la jerarquia

- La tabla de la clase resume bien la logica de diseño:
  - **registros**: mas rapidos, privados por hilo, vida corta.
  - **shared memory**: muy rapida, compartida dentro del bloque, vida del bloque.
  - **global memory**: gran capacidad, visible para todo el device, lenta.
  - **local memory**: privada por hilo pero lenta.
  - **constant/textures**: espacios especializados y cacheados.
- Esta jerarquia explica varias reglas practicas de CUDA:
  - dividir trabajo en bloques para poder cooperar via shared memory;
  - minimizar accesos repetidos a memoria global;
  - maximizar reutilizacion de datos;
  - pensar siempre en el alcance de los datos: thread, bloque o device.

### Evolucion arquitectonica: de Fermi a Turing

- Despues de G80, la clase recorre varias generaciones para mostrar que la arquitectura CUDA evoluciona, pero mantiene sus ideas base.
- La linea general es:
  - mas unidades de computo;
  - mejor jerarquia de memoria;
  - mayor eficiencia energetica;
  - soporte a nuevos tipos de operaciones;
  - mejoras en comunicacion y en precision numerica.

### Fermi

- **Fermi** se lanza en **2010**.
- Cambios importantes:
  - soporte completo del estandar IEEE 754-2008 para simple y doble precision;
  - unificacion del espacio de direcciones de memoria global, shared y local, mejorando soporte para C++;
  - **16 SMs** con **32 CUDA cores** cada uno, total **512 cores**;
  - **2 warp schedulers** por SM;
  - **64 KB on-chip** por SM, dividibles entre cache y shared memory en configuraciones `16K/48K` o `48K/16K`;
  - incorporacion de **cache L2** para acceso a memoria global;
  - mecanismos de deteccion y correccion de errores.
- Conceptualmente, Fermi muestra una evolucion importante:
  - la GPU sigue orientada a throughput y a muchos hilos;
  - pero se vuelve mas general, mas robusta y con una jerarquia de memoria mas sofisticada.
- Esto ayuda a entender que CUDA no es un modelo fijo:
  - conserva la logica de threads, blocks y warps;
  - pero el costo relativo de memoria y algunas optimizaciones cambia con la generacion.

### Kepler

- **Kepler** se lanza en **2012**.
- El foco principal es mejorar **performance por watt**.
- La idea de diseno es muy clara:
  - bajar frecuencia;
  - aumentar mucho la cantidad de cores.
- Ejemplo comparativo dado por la presentacion:
  - **Fermi GF110**: `512 CUDA cores`;
  - **Kepler GK104**: `1536 CUDA cores`.
- Otros cambios:
  - los nuevos multiprocesadores pasan a llamarse **SMX**;
  - aparecen **4 GPCs** con **2 SMXs** cada uno;
  - uso de **PCI Express 3.0** para comunicacion con el host;
  - cada SMX tiene **192 CUDA cores**, **32 SFUs** y **4 warp schedulers**;
  - se incorpora un cache de solo lectura de `48K`.
- Leccion de diseno:
  - el aumento de rendimiento no viene solo por hacer cada unidad mas rapida;
  - tambien viene por **replicar mas paralelismo util** con mejor eficiencia energetica.

### Maxwell

- **Maxwell** aparece a fines de **2014**.
- La presentacion destaca un salto en eficiencia energetica:
  - casi duplica la performance por watt respecto a Kepler.
- Caracteristicas remarcadas:
  - **4 GPCs**;
  - **16 multiprocesadores**;
  - **128 CUDA cores** por multiprocesador;
  - **2048 CUDA cores** en total.
- Los multiprocesadores se particionan en **4 bloques** con recursos propios de planificacion e instrucciones.
- Cada multiprocesador tiene **4 warp schedulers**, y cada scheduler puede despachar **dos instrucciones por warp por ciclo**.
- Conceptualmente, Maxwell refuerza la idea de especializar aun mas la organizacion interna para sostener throughput con menor consumo.

### Pascal

- **Pascal** se lanza en **abril de 2016**.
- La clase la presenta como un salto cualitativo importante.
- Cambios destacados:
  - mejora en la comunicacion **CPU-GPU** mediante **NVLink**;
  - aumento fuerte de bandwidth;
  - soporte real y util para **FP16**;
  - uso de **HBM2**, con gran aumento de bandwidth de memoria.
- La presentacion muestra para la **Tesla P100**:
  - `5.3 TFLOPS` en **FP64**;
  - `10.6 TFLOPS` en **FP32**;
  - `21.2 TFLOPS` en **FP16**.
- Tambien se destaca:
  - hasta **60 SMs**;
  - cada SM con **64 CUDA cores** y **4 Texture Units**;
  - gran capacidad total de cache L2;
  - mejora importante en doble precision, con ratio **2:1** entre SP y DP;
  - soporte IEEE 754-2008 completo, incluyendo **FMA** y desnormalizados.
- Idea conceptual:
  - Pascal muestra que la arquitectura CUDA ya no apunta solo a graficos o GPGPU general, sino tambien a cargas numericas intensivas y Deep Learning.
- La introduccion de **FP16** y mejor interconexion tambien explica por que ciertas aplicaciones modernas favorecen tipos de dato mas pequenos y mas transferencia efectiva.

### Volta

- **Volta** se lanza en **mayo de 2017**.
- Rendimientos mencionados:
  - `7.8 TFLOPS` en **FP64**;
  - `15.7 TFLOPS` en **FP32**;
  - `125 TFLOPS` en **FP16**.
- La gran novedad es la inclusion de **tensor cores**.
- Ademas:
  - se separan las unidades de **FP32** de las de **INT**;
  - se permite ejecutar simultaneamente operaciones enteras y de punto flotante.
- Cada tensor core puede realizar **64 FMA** por ciclo.
- Hay **8 tensor cores por multiprocesador**.
- La presentacion destaca mejoras muy fuertes en multiplicacion de matrices y en Deep Learning cuando se usan entradas FP16 y acumulacion FP32.
- Conceptualmente, Volta marca otra etapa del diseno:
  - ya no solo se optimiza el paralelismo masivo generico;
  - se agregan **unidades funcionales especializadas** para patrones de computo dominantes.

### Turing

- **Turing** aparece a fines de **2018**.
- La diapositiva de ejemplo usa la **GeForce RTX 2080 Ti**, con:
  - `4352 CUDA cores`;
  - `11 GB` de memoria;
  - `616 GB/s` de bandwidth;
  - `0.367 TFLOPS` en **FP64**;
  - `13.4 TFLOPS` en **FP32**;
  - `26.9 TFLOPS` en **FP16**;
  - `107.6 Tensor TFLOPS` con acumulacion en FP16.
- Mejoras destacadas:
  - incorporacion de **Ray Tracing cores**;
  - ejecucion concurrente de instrucciones de enteros y punto flotante;
  - nuevo diseno de **shared memory**;
  - tensor cores con soporte para `int8` e `int4`, utiles para **quantization**.
- Idea de fondo:
  - la arquitectura sigue ampliando el repertorio de unidades especializadas sin abandonar la base de SMs, warps y jerarquias de memoria.

### Tendencia general de las arquitecturas

- Las tablas comparativas muestran una tendencia clara:
  - crece mucho la cantidad de transistores;
  - crece mucho la cantidad de CUDA cores;
  - mejora mucho la **performance por watt**;
  - aumenta el **memory bandwidth**.
- Esta tendencia refuerza la tesis central del curso:
  - la mejora de rendimiento no vino por seguir aumentando frecuencia, sino por explotar aun mas paralelismo, ancho de banda y especializacion de hardware.
- Conexion con la clase 2:
  - es la misma razon historica por la que la computacion paralela se vuelve inevitable;
  - la GPU representa una respuesta extrema y especializada a ese problema.

### Compute capabilities

- Para usar bien una GPU no alcanza con saber que "es CUDA".
- Nvidia resume caracteristicas de sus GPUs mediante las **compute capabilities**.
- El formato usa dos numeros:
  - el primero indica cambio de generacion;
  - el segundo una revision.
- La clase recuerda:
  - las primeras GPUs CUDA eran **1.0**;
  - luego aparecen generaciones posteriores como Fermi, Kepler, Maxwell, Pascal, Volta y Turing;
  - tambien menciona arquitecturas mas recientes como **Ampere**, **Ada Lovelace**, **Hopper** y **Blackwell**.
- Importancia conceptual:
  - las compute capabilities indican que no todas las GPUs soportan exactamente los mismos limites, instrucciones ni recursos.
- La ultima tabla muestra ejemplos de diferencias entre generaciones en:
  - tamano de warp;
  - warps residentes por MP;
  - hilos residentes por MP;
  - cantidad de registros por MP;
  - dimension maxima del grid;
  - maximo de instrucciones por kernel.
- Idea de examen:
  - CUDA ofrece un modelo comun;
  - pero la optimizacion real depende de la generacion concreta del hardware.

### Que explica esta clase sobre la forma de programar CUDA

- Esta es la sintesis mas importante para estudiar:
  - se usan **muchos threads** porque la GPU necesita muchisimo paralelismo para ocultar latencia.
  - se usan **blocks** porque el bloque es la unidad natural de asignacion a un SM y de cooperacion mediante shared memory.
  - se usan **warps** como referencia de eficiencia porque son la unidad real de ejecucion y scheduling.
  - se evita la **divergencia** porque dentro de un warp obliga a serializar caminos.
  - se cuida la **jerarquia de memoria** porque no toda memoria tiene el mismo costo ni el mismo alcance.
  - se intenta maximizar trabajo independiente y regular porque el hardware esta pensado para throughput sostenido, no para control complejo por hilo.
- En continuidad con las clases anteriores:
  - la arquitectura de GPU materializa el **paralelismo de datos** de la clase 2;
  - vuelve criticos los problemas de **distribucion, sincronizacion y contencion** vistos en la clase 3;
  - y da la base fisica de por que CUDA tiene su modelo de programacion caracteristico.

## 4. Conceptos clave para memorizar

- La GPU CUDA se organiza jerarquicamente en **SMs** y unidades de calculo simples.
- El objetivo central del diseno es **maximizar throughput**, no minimizar la latencia de un solo hilo.
- La GPU oculta latencia alternando entre muchos hilos residentes.
- A nivel de programacion global, CUDA sigue el modelo **SPMD**.
- A nivel interno de cada multiprocesador, la ejecucion se organiza como **SIMT**.
- El programador piensa en `kernel -> grid -> blocks -> threads`, pero el hardware planifica en **warps**.
- **Warp** = grupo de 32 hilos en la arquitectura clasica presentada.
- La divergencia dentro de un warp serializa caminos y reduce eficiencia.
- **Shared memory** explica por que el bloque es la unidad natural de cooperacion.
- **Registros** son rapidos y privados por hilo.
- **Memoria local** es privada por hilo, pero lenta y off-chip.
- **Memoria global** tiene gran alcance, pero alto costo relativo.
- Las generaciones nuevas agregan:
  - mas cores;
  - mejor eficiencia energetica;
  - caches y jerarquias mas sofisticadas;
  - nuevas unidades especializadas como tensor cores y ray tracing cores.
- Las **compute capabilities** resumen diferencias funcionales y de recursos entre generaciones.

## 5. Posibles preguntas teoricas de examen

- Como esta organizada jerarquicamente una GPU CUDA desde la tarjeta hasta los CUDA cores.
- Por que la GPU necesita muchisimos hilos para lograr alto rendimiento.
- Que diferencia conceptual hay entre ocultar latencia con caches y ocultarla con cambio de contexto entre warps.
- Que significa que CUDA siga un modelo SPMD y que el hardware ejecute con SIMT.
- En que se diferencia SIMT de SIMD.
- Que es un warp y por que es la unidad relevante para entender eficiencia en CUDA.
- Por que la divergencia dentro de un warp perjudica el rendimiento.
- Por que los bloques deben ser relativamente independientes.
- Que relacion hay entre bloque, shared memory y cooperacion entre hilos.
- Cual es la diferencia entre memoria global, local, compartida, constante y registros.
- Por que la memoria local puede ser privada pero lenta.
- Que mejoras conceptuales introducen Fermi, Kepler, Maxwell, Pascal, Volta y Turing.
- Por que la evolucion de las GPUs refuerza la idea de paralelismo masivo y eficiencia energetica.
- Que son las compute capabilities y por que importan al optimizar un programa CUDA.

## 6. Dudas o ambiguedades detectadas en la presentacion, si las hay

- La presentacion usa el modelo "clasico" de memoria y ejecucion; varias caracteristicas cambian entre generaciones posteriores, por lo que conviene estudiar que se mantiene como idea general y que cambia como detalle de implementacion.
- La diapositiva que afirma que la GPU puede ejecutar un kernel a la vez simplifica el modelo historico presentado; para esta clase conviene tomarlo como parte del esquema introductorio, no como una afirmacion universal para toda generacion y todo contexto.
- En la tabla de texturas aparece `R??R-W??`, lo que sugiere una notacion poco clara sobre permisos de acceso; la idea segura para estudiar es que se trata de un espacio especializado y cacheado.
- La lista de compute capabilities mezcla generaciones antiguas con referencias mas nuevas; sirve como panorama evolutivo, pero no desarrolla en detalle que cambia en cada una.

## Contexto reutilizable sugerido

- Conviene reutilizar especialmente [Clase1-Introduccion-resumen.md](/Users/nicolaspereira/documents/facu/gpgpu/resumenes/Clase1-Introduccion-resumen.md) para mantener continuidad con:
  - `throughput vs latency`;
  - la comparacion conceptual entre CPU y GPU;
  - la motivacion historica de GPGPU.
- Conviene reutilizar [Clase2-Computacion_paralela-resumen.md](/Users/nicolaspereira/documents/facu/gpgpu/resumenes/Clase2-Computacion_paralela-resumen.md) para enlazar:
  - necesidad de paralelismo;
  - descomposicion de dominio;
  - balance de carga;
  - SIMD/SPMD como referencias conceptuales.
- Conviene reutilizar [Clase3-Programacion_paralela-resumen.md](/Users/nicolaspereira/documents/facu/gpgpu/resumenes/Clase3-Programacion_paralela-resumen.md) para reforzar:
  - divergencia;
  - variables privadas y compartidas;
  - reducciones;
  - sincronizacion y contencion.

## Nombre de archivo sugerido

- `Clase4-Arquitectura_de_la_GPU-resumen.md`

# Resumen Clase 2 - Computacion Paralela

## 1. Titulo del resumen

**Clase 2 - Computacion Paralela**

## 2. Temas principales

- Limites de la maquina secuencial de Von Neumann y necesidad de paralelismo.
- Formas de introducir paralelismo dentro de arquitecturas tradicionalmente secuenciales.
- Taxonomia de Flynn: SISD, SIMD, MISD y MIMD.
- Niveles de paralelismo y clasificaciones de programas paralelos.
- Modelos de programacion paralela: memoria compartida y memoria distribuida.
- Estrategias de descomposicion de problemas y criterios de particion.
- Paralelismo con hilos y OpenMP.
- Paralelismo distribuido con MPI.

## 3. Resumen desarrollado por secciones

### De la maquina secuencial al problema del rendimiento

- La clase parte de la **arquitectura de Von Neumann**, compuesta por:
  - una unidad funcional;
  - una unidad de control;
  - una memoria unica.
- En este modelo, un computo se realiza como una secuencia de pasos: buscar instruccion, decodificarla, calcular direcciones, traer datos, ejecutar la operacion y almacenar el resultado.
- La idea central es que este esquema describe bien a la computadora secuencial tradicional, pero tambien deja ver su limitacion principal: el trabajo avanza como una secuencia.

### Necesidad de paralelismo

- La presentacion remarca que la **Ley de Moore** sigue reflejando aumento en la cantidad de transistores, pero eso ya no se traduce linealmente en mayores frecuencias de reloj.
- La restriccion principal mencionada es **termica**: no se puede seguir aumentando indefinidamente la velocidad sin problemas de consumo y calor.
- Por eso, para mejorar rendimiento se vuelve necesario:
  - aumentar la cantidad de nucleos;
  - explotar paralelismo.
- Conexion con la clase 1:
  - En la introduccion, la GPU aparecia como hardware con muchisimos recursos dedicados a calculo y alto ancho de banda.
  - Esta clase da el marco general: **GPGPU no es una excepcion**, sino una respuesta extrema a la misma necesidad historica de aumentar rendimiento explotando paralelismo en lugar de solo subir frecuencia.

### Paralelismo dentro de maquinas secuenciales

- La clase muestra que incluso en una arquitectura de estilo Von Neumann se puede incorporar paralelismo de varias maneras:
  - **Multiples unidades funcionales / multi-core**: varios procesadores o nucleos dentro del mismo chip.
  - **Pipelining**: una tarea se divide en etapas, y distintas instrucciones pueden estar simultaneamente en distintas fases del pipeline.
  - **Instrucciones vectoriales**: una misma instruccion opera sobre varios datos a la vez, como en MMX, SSE o AVX.
  - **Paralelismo a nivel de memoria**: varias operaciones de memoria pueden quedar pendientes en simultaneo.
- Idea importante para examen:
  - No todo paralelismo significa "muchos programas corriendo". Tambien existe paralelismo interno al procesador, al compilador y a la jerarquia de memoria.
- Relacion con GPGPU:
  - Las extensiones vectoriales de CPU y el modelo SIMD anticipan conceptualmente parte de lo que luego se explota masivamente en GPU.
  - La diferencia es de escala y objetivo: la CPU agrega mecanismos de paralelismo sin abandonar su foco generalista y de baja latencia; la GPU organiza gran parte de su diseno alrededor del **throughput** y del trabajo masivo sobre datos.

### Jerarquia y organizacion de memoria

- La presentacion recuerda que la memoria es jerarquica y que el acceso a datos es un factor central del rendimiento.
- Tambien revisa como se almacenan arreglos y matrices:
  - los arreglos se almacenan de forma contigua;
  - en **C** las matrices suelen almacenarse por filas;
  - en **Fortran** se almacenan por columnas.
- Se muestran formulas para calcular la posicion de una celda en memoria.
- Idea conceptual:
  - La forma de recorrer datos no es un detalle menor; afecta localidad y costo de acceso.
- Relacion con la clase 1:
  - Si en la introduccion se destacaba el gran ancho de banda de memoria de la GPU, aca aparece el fundamento general: el paralelismo efectivo depende no solo del calculo, sino tambien de **como se organizan y acceden los datos**.

### Maquinas paralelas y taxonomia de Flynn

- La taxonomia de **Flynn** clasifica arquitecturas segun el flujo de instrucciones y el flujo de datos:
  - **SISD**: una instruccion, un dato. Es la maquina secuencial clasica.
  - **SIMD**: una instruccion, multiples datos. Varias unidades ejecutan la misma instruccion sobre datos distintos al mismo tiempo.
  - **MISD**: multiples instrucciones, un dato. Es poco comun en la practica.
  - **MIMD**: multiples instrucciones, multiples datos. Varias unidades autonomas trabajan sobre datos distintos con control descentralizado.
- Punto importante:
  - Esta clasificacion es conceptual; no toda arquitectura real encaja de forma absolutamente pura.
- Relacion explicita con GPGPU:
  - El modelo general visto en la clase 1, donde un **kernel** ejecuta el mismo calculo sobre muchos datos, se parece claramente a un enfoque **SIMD / data-parallel**.
  - Sin embargo, la GPU moderna tambien incorpora rasgos mas complejos que no se reducen a una etiqueta simple.
  - Para estudiar este curso, conviene quedarse con que **CUDA explota sobre todo paralelismo de datos**, mientras que esta clase presenta el mapa general completo de arquitecturas paralelas.

### Niveles de paralelismo

- La presentacion distingue distintos niveles:
  - **A nivel de trabajo**: multiprogramming, multiprocessing, manejado por el sistema operativo.
  - **A nivel de programa**: paralelismo dentro de una aplicacion; puede ser de grano fino o grueso.
  - **A nivel inter-instruccion**: explotado por compiladores mediante analisis de dependencias.
  - **A nivel intra-instruccion**: explotado por hardware, como pipelining.
- Esta clasificacion ayuda a separar responsabilidades:
  - algunas formas de paralelismo las maneja el hardware;
  - otras el compilador;
  - otras el sistema operativo;
  - otras el programador.
- Relacion con CUDA y GPGPU:
  - En CUDA, el programador trabaja sobre todo a **nivel de programa** y de **paralelismo de datos**, mientras el hardware de la GPU agrega ademas paralelismo interno propio.

### Como se clasifican los programas paralelos

- Los programas paralelos se pueden clasificar por:
  - **granularidad**: grano fino o grueso;
  - **homogeneidad**: misma tarea sobre distintos datos vs tareas diferentes;
  - **estrategia de paralelismo**: de control, de flujo o de datos;
  - **grado de programacion**: implicito o explicito.
- Punto de conexion con el curso:
  - El enfoque de GPGPU presentado en la clase 1 es, en general, un caso de:
  - paralelismo **homogeneo**;
  - de **datos**;
  - usualmente con una expresion bastante **explicita** por parte del programador.

### Modelos de programacion paralela

- La clase divide los grandes paradigmas en:
  - **memoria compartida**;
  - **memoria distribuida**.
- **Memoria compartida**:
  - todas las unidades acceden a una memoria comun;
  - hacen falta mecanismos de sincronizacion;
  - un ejemplo tipico es la programacion multi-hilo.
- **Memoria distribuida**:
  - cada procesador tiene memoria local;
  - la coordinacion se hace mediante pasaje de mensajes;
  - es comun en clusters, MPP y sistemas distribuidos.
- Idea importante:
  - El hardware condiciona fuertemente el modelo de programacion.
- Comparacion con la introduccion general del curso:
  - En la clase 1 se presento a la GPU como **device** coordinado por un **host**.
  - Esta clase amplia ese panorama mostrando que el curso se apoya en una forma particular de paralelismo, mientras que el campo general incluye modelos compartidos y distribuidos con costos distintos de comunicacion y sincronizacion.

### Descomposicion del problema

- Para paralelizar se aplica una estrategia de **divide and conquer**.
- La division puede ser:
  - **descomposicion funcional**: distintas funciones independientes para distintos procesadores.
  - **descomposicion de dominio**: misma funcion aplicada a distintas porciones de datos.
- Esta distincion es clave para entender por que muchos problemas encajan mejor que otros en GPU.
- Relacion explicita con GPGPU:
  - La GPU, tal como fue presentada en la clase 1, favorece claramente la **descomposicion de dominio**.
  - Es decir: tomar un gran conjunto de datos y aplicar el mismo kernel en paralelo.
  - La **descomposicion funcional** suele estar mas asociada a esquemas mas heterogeneos o a coordinacion entre componentes con trabajos diferentes.

### Criterios para una buena particion

- La presentacion destaca varios criterios para optimizar la particion del computo:
  - **balance de carga**;
  - minimizar comunicaciones y sincronizaciones;
  - minimizar ancho de banda usado;
  - minimizar la cantidad de nodos con los que hay que comunicarse.
- Tambien introduce el problema del **mapeo**:
  - como asociar unidades de computo con trabajo concreto;
  - como aprovechar cercania entre unidades y localidad de datos.
- Idea importante para examen:
  - Un programa paralelo correcto no necesariamente es eficiente.
  - Ademas de correctitud, importa mucho el costo de coordinacion.
- Conexion con la clase 1:
  - Esto refuerza por que en GPGPU no alcanza con "usar muchos hilos".
  - El beneficio aparece cuando el problema tiene suficiente independencia y buen acceso a datos como para que el paralelismo compense sus costos.

### Paralelismo de memoria compartida: hilos y OpenMP

- En memoria compartida, comunicaciones y sincronizaciones ocurren a traves del espacio comun de memoria.
- Hace falta garantizar **exclusion mutua** y controlar accesos concurrentes.
- **Hilos**:
  - un proceso puede tener varios hilos de ejecucion;
  - los hilos comparten el mismo espacio de direcciones;
  - esto permite paralelismo dentro del proceso.
- Ventajas de hilos:
  - intercambio de informacion mas eficiente que con procesos separados;
  - mayor control sobre el paralelismo.
- Desventajas:
  - exigen mas trabajo del programador;
  - no escalan mas alla de una sola maquina.
- **OpenMP**:
  - API portable para C/C++ y Fortran;
  - usa directivas de compilador;
  - pierde algo de eficiencia y flexibilidad frente a soluciones mas manuales, pero gana en portabilidad.
- Modelo de ejecucion de OpenMP:
  - **fork-join**: un hilo maestro crea una region paralela, reparte trabajo y luego continua solo al terminar.
- Las sincronizaciones en OpenMP son necesarias para evitar conflictos sobre variables compartidas, pero son costosas.
- Relacion con el enfoque general del curso:
  - OpenMP representa una forma clasica de paralelismo sobre CPU con memoria compartida.
  - CUDA, en cambio, trabaja sobre un acelerador con un modelo distinto, aunque comparte problemas conceptuales como reparto de trabajo, sincronizacion y organizacion de datos.

### Paralelismo de memoria distribuida: MPI

- Para memoria distribuida, la presentacion menciona **MPI (Message Passing Interface)** como estandar principal.
- MPI:
  - se basa en **pasaje de mensajes**;
  - define sintaxis y semantica de rutinas para programas paralelos;
  - funciona sobre diversas arquitecturas;
  - permite programar usando C, Fortran y otros lenguajes tradicionales.
- En este modelo:
  - cada proceso es un programa secuencial separado;
  - los datos son privados a cada proceso;
  - la comunicacion debe hacerse explicitamente.
- La presentacion lo asocia a un esquema **SPMD (Single Program Multiple Data)**.
- Relacion con GPGPU:
  - MPI y CUDA atacan problemas de paralelismo distintos.
  - MPI organiza multiples procesos que cooperan mediante mensajes; CUDA organiza una gran cantidad de hilos o unidades de trabajo sobre un acelerador.
  - Aun asi, ambos comparten la idea central de este curso y de la clase 1: **el rendimiento extra exige estructurar el problema para explotar concurrencia real**.

## 4. Conceptos clave para memorizar

- La limitacion del aumento de frecuencia obliga a mejorar rendimiento explotando paralelismo.
- La arquitectura de Von Neumann es el punto de partida del modelo secuencial clasico.
- El paralelismo puede existir a varios niveles: trabajo, programa, inter-instruccion e intra-instruccion.
- Formas de paralelismo en maquinas secuenciales: multi-core, pipelining, instrucciones vectoriales y paralelismo de memoria.
- Taxonomia de Flynn:
  - `SISD`: secuencial clasico.
  - `SIMD`: misma instruccion sobre muchos datos.
  - `MISD`: raro en la practica.
  - `MIMD`: instrucciones y datos multiples, control descentralizado.
- Dos grandes modelos de programacion paralela:
  - memoria compartida;
  - memoria distribuida.
- Dos formas de descomponer problemas:
  - funcional;
  - de dominio.
- Criterios de eficiencia paralela:
  - balance de carga;
  - minimizar comunicacion;
  - minimizar sincronizacion;
  - aprovechar localidad de datos.
- OpenMP es un modelo de memoria compartida con hilos y ejecucion fork-join.
- MPI es un estandar de memoria distribuida basado en pasaje de mensajes y usualmente en SPMD.
- Conexion con la clase 1:
  - GPGPU y CUDA se entienden mejor como un caso particular de computacion paralela fuertemente orientado a **paralelismo de datos**, **descomposicion de dominio** y **throughput**.

## 5. Posibles preguntas teoricas de examen

- Por que el aumento de transistores ya no alcanza por si solo para mejorar rendimiento.
- Cuales son las limitaciones del modelo secuencial de Von Neumann.
- Que formas de paralelismo pueden incorporarse dentro de una maquina secuencial.
- Que diferencia hay entre pipelining, multi-core e instrucciones vectoriales.
- En que consiste la taxonomia de Flynn y que caracteriza a SISD, SIMD, MISD y MIMD.
- Que niveles de paralelismo existen y quien suele explotarlos.
- Que diferencia hay entre grano fino y grano grueso.
- Que diferencia hay entre memoria compartida y memoria distribuida.
- Que es la descomposicion funcional y que es la descomposicion de dominio.
- Por que comunicacion, sincronizacion y balance de carga son claves en programas paralelos.
- Cuales son las ventajas y desventajas del uso de hilos.
- Como funciona el modelo fork-join de OpenMP.
- Que es MPI y por que se asocia a memoria distribuida.
- Como se relaciona el enfoque de CUDA/GPGPU con el paralelismo SIMD o con la descomposicion de dominio.

## 6. Dudas o ambiguedades detectadas en la presentacion, si las hay

- Las diapositivas sobre jerarquia de memoria son mayormente graficas, por lo que la idea general es clara pero no se detallan formalmente todos los niveles ni sus costos relativos.
- La relacion exacta entre la taxonomia de Flynn y las GPUs modernas no se desarrolla en profundidad; la vinculacion con SIMD se puede hacer de forma conceptual, pero sin asumir que toda GPU moderna encaja de manera simple en una sola categoria.
- La presentacion nombra varios criterios de optimizacion de particion, pero no los desarrolla con ejemplos cuantitativos; conviene estudiarlos como principios generales.

## Contexto reutilizable sugerido

- Conviene reutilizar especialmente [Clase1-Introduccion-resumen.md](/Users/nicolaspereira/documents/facu/gpgpu/resumenes/Clase1-Introduccion-resumen.md) para mantener continuidad en:
  - `GPU`, `GPGPU`, `CUDA`, `host`, `device`, `kernel`;
  - la distincion entre **throughput** y **latency**;
  - la idea de que la GPU se justifica cuando hay muchas operaciones similares e independientes.
- Este resumen de clase 2 deberia reutilizarse luego en clases de programacion CUDA y arquitectura, porque introduce el vocabulario general de computacion paralela sobre el que despues se monta el modelo de GPU.

## Nombre de archivo sugerido

- `Clase2-Computacion_paralela-resumen.md`

# Resumen Clase 5 - Programacion CUDA

## 1. Titulo del resumen

**Clase 5 - Programacion CUDA**

## 2. Temas principales

- CUDA como modelo de programacion para usar la GPU como coprocesador.
- Separacion conceptual entre **host** (CPU) y **device** (GPU).
- Flujo basico de un programa CUDA: reservar, transferir, ejecutar y recuperar.
- Organizacion jerarquica de la ejecucion: **grid -> blocks -> threads**.
- Identidad de cada hilo y mapeo entre hilos y datos.
- Cooperacion dentro del bloque y limites de cooperacion entre bloques.
- Relacion entre espacios de memoria y alcance de los datos.
- Ejemplos de suma por filas como estudio de distribucion de trabajo, escalabilidad y race conditions.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase introduce **CUDA** como el modelo que permite expresar computo masivamente paralelo sobre GPU.
- Conexion con los resumenes anteriores:
  - la **clase 3** habia introducido los problemas de programacion paralela: reparto de trabajo, sincronizacion, reducciones y race conditions;
  - la **clase 4** habia mostrado la arquitectura GPU: muchos hilos, bloques, warps y jerarquia de memoria;
  - la **clase 5** traduce esas ideas a un modelo de programacion concreto.
- Idea clave:
  - CUDA no es solo "escribir codigo para GPU";
  - es una forma de **describir miles de tareas similares** para que la GPU las distribuya sobre su hardware.

### GPU como device y CPU como host

- La presentacion parte de una separacion fundamental:
  - la **CPU** actua como **host**;
  - la **GPU** actua como **device** o coprocesador.
- La GPU:
  - tiene su propia memoria del device;
  - ejecuta muchisimos threads livianos en paralelo;
  - se usa para la parte del calculo que tiene **paralelismo de datos**.
- Esta idea conecta directamente con la clase 4:
  - la GPU esta construida para throughput, no para minimizar la latencia de un unico hilo;
  - por eso CUDA intenta mover al device las partes del problema que pueden descomponerse en muchas tareas semejantes.

### Flujo general de un programa CUDA

- El programa CUDA integra codigo del host con codigo que corre en la GPU.
- El flujo basico que muestra la clase es:
  1. el host ejecuta instrucciones normales;
  2. reserva memoria en la GPU;
  3. copia datos del host al device;
  4. lanza un **kernel** para procesarlos;
  5. recupera resultados al host;
  6. el host continua la ejecucion.
- Conceptualmente, esto deja una idea importante para examen:
  - la GPU no reemplaza a la CPU;
  - la CPU **orquesta** y la GPU **acelera** las partes adecuadas.
- Tambien deja claro que el costo de mover datos existe.
- Por eso, usar GPU tiene sentido cuando el trabajo paralelo compensa el costo de transferencia.

### Threads de GPU vs threads de CPU

- La clase compara explicitamente los hilos de CPU con los de GPU.
- En CPU:
  - los hilos son relativamente costosos;
  - suelen usarse pocos, en el orden de los cores disponibles.
- En GPU:
  - los hilos son mucho mas livianos;
  - se necesitan **miles** para lograr buen rendimiento.
- Esta diferencia no es accidental:
  - en CPU se busca que cada hilo sea potente;
  - en GPU se busca que existan muchos hilos listos para ocultar latencia, tal como se explico en la clase 4.
- Consecuencia conceptual:
  - en CUDA conviene pensar en tareas pequenas, repetibles y numerosas;
  - no en pocos hilos con logica compleja.

### Kernels: una misma rutina para muchos datos

- Las rutinas que corren en GPU se llaman **kernels**.
- Un kernel representa una funcion que se ejecuta muchas veces en paralelo.
- Todos los threads ejecutan esencialmente el mismo codigo, pero cada uno:
  - tiene una identidad propia;
  - trabaja sobre una porcion distinta de los datos;
  - puede tomar decisiones segun ese identificador.
- Este es el patron fundamental de CUDA:
  - **mismo programa, multiples hilos, distintos datos**.
- La presentacion escribe "SPMT", pero conceptualmente coincide con la idea vista antes de **SPMD**: muchos hilos ejecutan el mismo programa sobre datos distintos.

### Jerarquia de ejecucion: grid, blocks y threads

- CUDA organiza la ejecucion en tres niveles:
  - **threads**;
  - **blocks**;
  - **grid**.
- Los threads se agrupan en bloques, y los bloques forman una grilla.
- Esta jerarquia no es decorativa:
  - permite describir mucho paralelismo potencial;
  - le deja al hardware decidir como mapearlo a los multiprocesadores reales.
- La regla conceptual mas importante es:
  - **los bloques deben ser independientes entre si**.
- Motivo:
  - distintos bloques pueden ejecutarse en cualquier orden;
  - pueden correr en multiprocesadores distintos;
  - no se debe asumir sincronizacion ni comunicacion directa entre ellos.
- Esto extiende una idea ya instalada en la clase 4:
  - la escalabilidad de CUDA surge porque el programador define el trabajo total, pero no controla el orden exacto en que se asigna cada bloque al hardware.

### Por que los bloques importan tanto

- La clase enfatiza que threads del mismo bloque pueden cooperar.
- Esa cooperacion ocurre mediante:
  - **memoria shared**;
  - **operaciones atomicas**;
  - **barreras de sincronizacion**.
- En cambio, threads de bloques distintos no cooperan directamente.
- Esta restriccion es clave para entender CUDA:
  - el bloque es la unidad natural de **cooperacion local**;
  - el grid es la unidad de **paralelismo global**.
- Conexion con la clase 3:
  - los problemas de variables privadas, variables compartidas y sincronizacion reaparecen ahora con una frontera concreta: thread, bloque o device.

### Identidad del hilo y mapeo a los datos

- Cada hilo tiene identificadores que permiten saber:
  - en que bloque esta;
  - cual es su posicion dentro del bloque;
  - cuantas dimensiones tienen el bloque y la grilla.
- La presentacion remarca que este esquema puede ser 1D, 2D o 3D.
- La idea importante no es memorizar nombres, sino la intuicion:
  - el programador **mapea indices logicos a datos**.
- Esto es especialmente util en problemas donde los datos ya tienen estructura espacial:
  - imagenes;
  - matrices;
  - grillas regulares;
  - dominios 3D.
- Idea de examen:
  - CUDA hace natural expresar **descomposicion de dominio** porque la jerarquia de indices se parece a la forma del problema.

### Espacios de memoria y alcance

- La clase vuelve sobre los espacios de memoria desde el punto de vista del programador:
  - memoria del host;
  - memoria global del device;
  - memoria compartida;
  - memoria constante;
  - variables privadas por hilo.
- La idea clave para estudiar no es la sintaxis, sino el **alcance**:
  - algunos datos pertenecen al host;
  - otros a todo el device;
  - otros a un bloque;
  - otros a un hilo individual.
- Conexion directa con la clase 4:
  - **registros** y estado privado por hilo;
  - **shared memory** para cooperacion rapida dentro del bloque;
  - **global memory** para datos visibles por todo el device pero mas costosos;
  - **constant memory** como espacio especializado de solo lectura.
- Idea conceptual fuerte:
  - en CUDA, decidir **donde vive** un dato es parte del algoritmo, no un detalle secundario.

### Reserva, transferencia y lanzamiento

- La presentacion muestra funciones para:
  - reservar memoria en el device;
  - copiar datos entre host y device;
  - lanzar kernels;
  - sincronizar host con GPU.
- Para el examen importa sobre todo esta lectura conceptual:
  - el host administra la memoria global del device;
  - la GPU no opera magicamente sobre las variables del host;
  - hay una frontera explicita entre ambos espacios.
- Tambien aparece una sincronizacion explicita del host con el device.
- Esto es importante porque muestra que:
  - lanzar un kernel no equivale automaticamente a "ya termino";
  - la coordinacion host-device forma parte del modelo de ejecucion.

### Sincronizacion: dos niveles distintos

- La clase menciona dos formas de sincronizacion con papeles diferentes:
  - sincronizacion entre **host y device**;
  - sincronizacion entre **threads de un mismo bloque**.
- Esta distincion es muy importante:
  - una coordina etapas grandes del programa;
  - la otra coordina cooperacion local dentro del kernel.
- Conexion con la clase 3:
  - vuelve la idea de que sincronizar es necesario para correctitud;
  - pero sincronizar demasiado o mal puede limitar el paralelismo efectivo.

### Operaciones atomicas

- La clase reintroduce las **atomicas** en un contexto ya claramente CUDA.
- Conceptualmente, una atomica sirve cuando varios hilos quieren actualizar una misma posicion de memoria sin perder actualizaciones.
- Esto resuelve un problema de correctitud, pero no elimina el costo:
  - sigue habiendo contencion sobre esa direccion;
  - parte del trabajo se serializa en ese punto.
- Idea de examen:
  - una atomica **protege** una actualizacion compartida;
  - no convierte automaticamente una solucion en eficiente.

### Ejemplo guia: suma por filas de una matriz

- La segunda mitad de la clase usa repetidamente el problema de **sumar por filas una matriz**.
- El valor teorico del ejemplo no esta en el codigo, sino en como va mostrando distintas decisiones de mapeo.

### Primera version: un hilo por fila dentro de un solo bloque

- En la primera version, cada hilo suma una fila completa.
- Conceptualmente:
  - el mapeo entre hilos y datos es simple;
  - cada hilo trabaja de forma independiente;
  - no hace falta cooperacion entre hilos.
- Pero tiene dos limites claros:
  - solo sirve mientras la cantidad de filas entre en un bloque;
  - usa un solo bloque, asi que desaprovecha la posibilidad de distribuir trabajo entre varios multiprocesadores.
- Leccion:
  - una solucion puede ser correcta y simple, pero no **escalable**.

### Segunda version: varios bloques

- La siguiente version reparte las filas entre varios bloques.
- La idea importante es que el indice del hilo ya no puede pensarse solo dentro del bloque:
  - hay que combinar posicion del bloque y posicion interna del hilo para obtener un identificador global de trabajo.
- Conceptualmente, esta es una de las transiciones centrales al programar en CUDA:
  - pasar de "quien soy dentro del bloque" a "que parte del problema global me toca resolver".
- Esta version ya respeta mejor la arquitectura vista en la clase 4:
  - mas bloques;
  - mas oportunidad de ocupacion del hardware;
  - mejor escalabilidad.

### Tercera version: repartir una misma fila entre varios hilos

- Luego la clase plantea otro caso:
  - si hay pocas filas pero muchas columnas, asignar una fila entera a un solo hilo desaprovecha paralelismo.
- Entonces cambia la estrategia:
  - varios hilos colaboran para sumar distintas partes de la misma fila.
- Esto es teoricamente muy valioso porque muestra que la descomposicion depende de la forma de los datos:
  - si el paralelismo no esta en la cantidad de filas, puede estar en las columnas;
  - el mapeo optimo depende de donde esta el trabajo.
- Conexion con clase 3:
  - reaparece el problema de **distribucion de datos/calculos** y **balance de carga**.

### El problema de correctitud en el ejemplo

- Cuando varios hilos aportan parciales a la misma suma por fila, aparece un clasico problema de concurrencia:
  - varios escriben sobre el mismo resultado.
- La clase marca explicitamente la **race condition**.
- Esto es importante porque muestra que:
  - aumentar el paralelismo suele exigir luego un mecanismo de coordinacion;
  - mas hilos no implica automaticamente solucion correcta.
- La respuesta mostrada es usar una **atomicAdd** para combinar parciales.
- Esto enlaza perfectamente con la clase 3:
  - ya no es una race condition abstracta;
  - ahora aparece en un kernel CUDA concreto.

### Distribucion por bloques vs distribucion ciclica

- La clase recupera explicitamente dos estrategias ya vistas:
  - **distribucion por bloques**;
  - **distribucion ciclica**.
- En el ejemplo de suma por filas, ambas sirven para repartir columnas entre threads.
- Conceptualmente:
  - la distribucion por bloques da a cada hilo un segmento contiguo;
  - la distribucion ciclica intercala elementos entre hilos.
- La importancia teorica es doble:
  - afectan el **balance de carga**;
  - afectan el **patron de acceso a memoria**.
- Aunque la clase solo lo anticipa, deja una idea importante:
  - en GPU no solo importa cuanto trabajo recibe cada hilo;
  - tambien importa **como accede a memoria** al hacerlo.

### Correctitud no alcanza: tambien importa el acceso a memoria

- La clase senala que una version corregida con atomicas sigue teniendo problemas de eficiencia.
- En particular, menciona que el patron de acceso a memoria no es bueno para GPU.
- Esta observacion es muy importante conceptualmente:
  - una solucion puede ser correcta;
  - una solucion puede incluso explotar mas paralelismo;
  - y aun asi puede rendir mal si el acceso a memoria no acompana la arquitectura.
- Esto prepara el terreno para clases posteriores:
  - optimizar CUDA no es solo lanzar muchos hilos;
  - tambien es organizar bien la cooperacion y los accesos a memoria.

### Que deja instalada esta clase

- Esta clase fija el modelo mental basico para programar en CUDA:
  - el host organiza;
  - la GPU ejecuta kernels;
  - el trabajo se expresa como muchos threads;
  - los threads se organizan en bloques y grids;
  - la cooperacion fuerte ocurre dentro del bloque;
  - la correctitud depende de manejar bien accesos compartidos;
  - el rendimiento depende tanto del reparto de trabajo como del uso de memoria.
- En otras palabras:
  - la clase 4 explicaba **por que** la GPU esta organizada asi;
  - la clase 5 muestra **como pensar programas** que encajen en esa organizacion.

## 4. Conceptos clave para memorizar

- **CUDA** = modelo de programacion para usar la GPU como acelerador de computo masivamente paralelo.
- **Host** = CPU que controla la ejecucion general.
- **Device** = GPU que ejecuta kernels y posee su propia memoria.
- **Kernel** = rutina que se lanza para ejecutarse con muchos hilos en paralelo.
- En GPU se usan **miles de hilos livianos**, no pocos hilos pesados como en CPU.
- Jerarquia de ejecucion:
  - **thread** = unidad logica minima de trabajo;
  - **block** = grupo de threads que puede cooperar;
  - **grid** = conjunto de bloques que forman una ejecucion del kernel.
- Los **bloques deben ser independientes** porque pueden ejecutarse en cualquier orden.
- La cooperacion directa existe **dentro del bloque**, no entre bloques.
- La identidad del hilo sirve para mapear trabajo y datos.
- Los espacios de memoria se entienden por **alcance**: host, device, bloque o hilo.
- Programar en CUDA implica gestionar:
  - reserva de memoria en device;
  - transferencias host-device;
  - lanzamiento del kernel;
  - sincronizacion.
- **Race condition** = varios hilos actualizan un mismo dato sin coordinacion suficiente.
- **Atomicas** = solucionan correctitud en actualizaciones compartidas puntuales, pero pueden introducir contencion.
- Una solucion CUDA debe ser:
  - correcta;
  - escalable;
  - coherente con el patron de acceso a memoria de la GPU.

## 5. Posibles preguntas teoricas de examen

- Que problema resuelve CUDA y por que se adapta bien a la arquitectura GPU.
- Cual es la diferencia conceptual entre **host** y **device**.
- Cual es el flujo basico de ejecucion de un programa CUDA.
- Por que en GPU se usan miles de hilos y no pocos hilos como en CPU.
- Que funcion cumple un **kernel** dentro del modelo CUDA.
- Como se organiza la ejecucion en **threads, blocks y grid**.
- Por que los bloques deben ser independientes entre si.
- Que tipos de cooperacion son posibles dentro de un bloque.
- Por que la identidad del hilo es central en CUDA.
- Como se relaciona la jerarquia de ejecucion con la descomposicion de dominio.
- Que diferencia conceptual hay entre sincronizacion host-device y sincronizacion entre threads del bloque.
- Que es una **race condition** en CUDA y como aparece en el ejemplo de suma por filas.
- Para que sirve una operacion atomica y por que no garantiza buen rendimiento.
- Por que una solucion correcta puede seguir siendo mala en GPU desde el punto de vista del acceso a memoria.
- En que sentido la clase 5 extiende las ideas de programacion paralela de la clase 3 y de arquitectura GPU de la clase 4.

## 6. Dudas o ambiguedades detectadas en la presentacion

- La presentacion usa la sigla **SPMT**, pero conceptualmente parece referirse al modelo de muchos hilos ejecutando el mismo programa sobre datos distintos, en continuidad con la idea de **SPMD**.
- Algunas diapositivas finales del ejemplo de suma por filas parecen mas esquematicas que completas y tienen pequenos detalles de notacion o indices que no quedan del todo prolijos.
- La clase avisa que el patron de acceso a memoria del ejemplo no es bueno para GPU, pero no desarrolla todavia el criterio arquitectonico exacto; probablemente eso se profundiza en clases practicas o posteriores.

## Resumenes anteriores que conviene reutilizar como contexto

- [Clase3-Programacion_paralela-resumen.md](/Users/nicolaspereira/Documents/facu/gpgpu/resumenes/Clase3-Programacion_paralela-resumen.md) porque introduce distribucion de trabajo, race conditions, sincronizacion, reducciones y atomicas.
- [Clase4-Arquitectura_de_la_GPU-resumen.md](/Users/nicolaspereira/Documents/facu/gpgpu/resumenes/Clase4-Arquitectura_de_la_GPU-resumen.md) porque explica por que CUDA usa muchos hilos, bloques, warps y distintos espacios de memoria.

## Nombre de archivo propuesto

- `resumenes/Clase5-Programacion_CUDA-resumen.md`

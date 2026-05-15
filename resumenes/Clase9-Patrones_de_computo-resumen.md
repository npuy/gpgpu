# Resumen Clase 9 - Patrones de computo

## 1. Titulo del resumen

**Clase 9 - Patrones de computo I**

## 2. Temas principales

- Patrones de computo en CUDA como estructuras reutilizables para resolver familias de problemas paralelos.
- Histograma como patron con actualizaciones concurrentes sobre un conjunto acotado de cubetas.
- Importancia del acceso **coalesced** al leer la entrada.
- Operaciones **atomicas** como mecanismo de correctitud ante condiciones de carrera.
- **Privatization** como tecnica para reducir contencion y serializacion.
- **Reduce** como patron para combinar muchos valores en un resultado agregado.
- Optimizaciones sucesivas de reduccion basadas en memoria compartida, control de divergencia y conflictos de bancos.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase introduce los **patrones de computo** como soluciones recurrentes para problemas tipicos en GPGPU.
- La idea no es presentar kernels aislados, sino mostrar que ciertos problemas aparecen una y otra vez con la misma estructura:
  - muchos elementos de entrada;
  - trabajo repartido entre hilos;
  - una forma de combinacion o acumulacion;
  - cuellos de botella previsibles de memoria, sincronizacion o contencion.
- En esta primera parte aparecen dos patrones clasicos:
  - **histograma**;
  - **reduccion**.
- Ambos conectan directamente con clases anteriores:
  - de la **clase 5**, retoman la distribucion `grid -> blocks -> threads`;
  - de la **clase 6**, retoman coalescing, shared memory, atomicas y bank conflicts;
  - de la **clase 7**, conservan la idea de cooperacion controlada entre hilos;
  - de la **clase 8**, heredan la necesidad de medir y depurar optimizaciones en lugar de asumir que una version "mas paralela" necesariamente rinde mejor.

### Que significa estudiar un patron de computo

- Un patron de computo es una **forma reusable de organizar trabajo y datos** para una familia de problemas.
- No define solo "que se calcula", sino tambien:
  - como se particiona la entrada;
  - como cooperan los hilos;
  - donde aparece la contencion;
  - que recurso de hardware domina el costo.
- La ventaja de pensar en patrones es conceptual:
  - permite reconocer rapidamente que optimizaciones y riesgos son esperables.
- Idea importante para examen:
  - en GPU no alcanza con identificar paralelismo;
  - tambien hay que identificar **el tipo de interaccion entre hilos** que el problema exige.

### Patron 1: histograma

- El histograma toma muchos valores de entrada y los clasifica en un conjunto de **cubetas** o bins.
- Cada elemento de entrada contribuye a **incrementar un contador** asociado a su categoria.
- La presentacion usa un ejemplo con texto:
  - se definen rangos de letras;
  - cada caracter incrementa la cubeta correspondiente.
- Como patron, el histograma aparece cuando:
  - hay que construir distribuciones de frecuencia;
  - varios elementos pueden contribuir al mismo destino;
  - la salida tiene menos posiciones que la entrada.
- El problema conceptual que resuelve es:
  - transformar una gran cantidad de observaciones individuales en una **estructura agregada de conteos**.

### Primera idea de paralelizacion del histograma

- La primera aproximacion parte la entrada en **secciones**.
- Cada hilo procesa secuencialmente una seccion propia.
- Es una estrategia natural porque:
  - reparte trabajo de manera simple;
  - mantiene independencia al leer la entrada;
  - parece aprovechar muchos hilos.
- Pero la clase muestra que esta version tiene un problema importante:
  - los hilos adyacentes no leen posiciones adyacentes al mismo tiempo;
  - por lo tanto, los accesos a memoria global no son **coalesced**.
- Conexion con la clase 6:
  - el patron de acceso a la entrada importa tanto como la cantidad de hilos.

### Acceso intercalado en histograma

- La mejora propuesta es un acceso **intercalado** o strided.
- En vez de que cada hilo agote una seccion contigua por su cuenta, todos los hilos avanzan coordinadamente sobre tramos contiguos de la entrada.
- Conceptualmente:
  - el conjunto del warp o del bloque cubre una zona contigua;
  - luego todos avanzan al siguiente tramo.
- El problema que resuelve:
  - mejora el uso del ancho de banda al favorecer accesos **coalesced**.
- Supuesto implicito:
  - conviene reorganizar quien procesa cada dato si eso mejora el patron colectivo de acceso.
- Trade-off:
  - el trabajo total no cambia, pero cambia el orden de recorrido para alinearse mejor con el hardware.

### El verdadero cuello de botella del histograma: las actualizaciones concurrentes

- Aunque leer la entrada pueda hacerse eficientemente, el histograma tiene un problema estructural:
  - muchos hilos pueden querer actualizar la **misma cubeta**.
- Esa situacion genera una clasica operacion **read-modify-write**:
  1. leer el contador actual;
  2. sumarle uno;
  3. escribirlo de nuevo.
- Si dos o mas hilos hacen eso sobre la misma direccion sin proteccion, aparecen **condiciones de carrera**.
- El problema no es solo de rendimiento:
  - es primero un problema de **correctitud**.
- La clase muestra justamente que el valor final puede depender del orden relativo de ejecucion.

### Operaciones atomicas como solucion de correctitud

- Las **atomicas** convierten la secuencia leer-modificar-escribir en una unica operacion indivisible sobre una direccion.
- El problema que resuelven:
  - garantizan que no se pierdan actualizaciones cuando varios hilos modifican la misma posicion.
- Idea conceptual clave:
  - las atomicas no eliminan la contencion;
  - la vuelven **correcta**.
- Cuando muchos hilos compiten por la misma cubeta:
  - el hardware serializa esas actualizaciones en esa direccion.
- Trade-off central:
  - se gana correctitud, pero se puede perder paralelismo efectivo en el punto de acumulacion.
- Esto extiende lo visto en la clase 5:
  - una actualizacion compartida protegida puede seguir siendo un cuello de botella aunque el kernel sea correcto.

### Donde se ejecutan las atomicas y por que importa

- La presentacion distingue tres contextos:
  - atomicas sobre **memoria global**;
  - atomicas que pegan en **cache L2**;
  - atomicas sobre **memoria compartida**.
- La idea conceptual no es memorizar latencias exactas, sino el orden relativo:
  - global es la opcion mas costosa;
  - L2 mejora frente a RAM;
  - shared memory es mucho mas rapida, pero queda restringida al bloque.
- Esto deja un principio general:
  - en patrones con acumulacion concurrente, la ubicacion del acumulador importa tanto como la operacion usada.

### Privatization en histogramas

- La **privatizacion** aparece como la tecnica principal para reducir el costo de la contencion.
- Idea general:
  - en lugar de que todos los hilos actualicen una sola copia global del histograma,
  - se crean **copias privadas o parciales**;
  - cada grupo acumula localmente;
  - al final se fusionan esas copias en un resultado final.
- El problema que resuelve:
  - reduce la serializacion causada por demasiadas atomicas sobre las mismas direcciones.
- Esto encaja muy bien con CUDA porque:
  - la cooperacion natural existe a nivel de bloque;
  - la memoria compartida permite construir acumuladores locales de baja latencia.

### Supuestos y trade-offs de la privatizacion

- Supuesto principal:
  - el costo extra de mantener varias copias y luego fusionarlas es menor que el costo de la contencion directa.
- Ventajas:
  - menos contencion;
  - menor latencia por actualizacion local;
  - mejor escalabilidad cuando muchas entradas caen en pocas cubetas.
- Costos conceptuales:
  - mas uso de memoria;
  - una fase adicional de combinacion;
  - mayor complejidad de diseño.
- En otras palabras:
  - la privatizacion intercambia **redundancia temporal o espacial** por menor serializacion.

### Lectura integradora del patron histograma

- El histograma enseña una leccion muy importante:
  - hay patrones donde la entrada se procesa de forma muy paralela,
  - pero la salida concentra actualizaciones y obliga a controlar conflicto.
- El diseño correcto pasa por separar dos preguntas:
  1. como leer la entrada de forma eficiente;
  2. como acumular resultados sin perder correctitud ni colapsar por contencion.
- Para examen, conviene verlo como un patron de:
  - **muchos productores**;
  - **pocos destinos compartidos**.

### Patron 2: reduce

- La **reduccion** toma muchos valores y los combina en uno o pocos resultados agregados.
- El caso mostrado es la **suma** paralela.
- Como patron, reduce aparece cuando:
  - se quiere sumar, contar, maximizar, minimizar o combinar una coleccion;
  - la operacion de combinacion es asociativa o puede organizarse jerarquicamente.
- El problema que resuelve:
  - convertir una gran cantidad de datos independientes en un resumen agregado explotando paralelismo parcial.
- La presentacion lo trabaja como reduccion por bloques:
  - cada bloque produce un subtotal;
  - luego esos subtotales pueden seguir reduciendose en GPU o terminarse en CPU.

### Estructura conceptual de una reduccion en GPU

- La reduccion se apoya en una idea distinta a la del histograma:
  - en vez de muchas actualizaciones concurrentes a destinos arbitrarios,
  - se organiza una **combinacion cooperativa y jerarquica**.
- El esquema general es:
  1. cargar datos desde memoria global a **shared memory**;
  2. sincronizar;
  3. combinar pares de valores en etapas sucesivas;
  4. dejar un subtotal por bloque.
- Conexion con la clase 6:
  - esto es un uso tipico de shared memory como espacio de reutilizacion y cooperacion local.

### Primer enfoque de reduccion

- El primer kernel hace la reduccion por etapas usando `step = 1, 2, 4, ...`.
- El patron base es correcto:
  - cada etapa reduce la cantidad de elementos activos;
  - al final queda un solo resultado por bloque.
- Sin embargo, la clase marca un problema fuerte:
  - **divergencia de warps**.
- Motivo conceptual:
  - en cada iteracion solo algunos hilos realizan trabajo util y otros quedan inactivos segun una condicion modular.
- Problema que muestra este enfoque:
  - una reduccion puede tener paralelismo aparente, pero organizarlo mal genera ineficiencia en la ejecucion SIMT.

### Segundo enfoque de reduccion

- La segunda version reorganiza la condicion de participacion.
- Sigue habiendo reduccion por etapas, pero cambia la forma en que se seleccionan los hilos activos.
- El problema que resuelve:
  - disminuye la divergencia respecto de la primera version.
- La presentacion muestra una mejora importante de tiempo de ejecucion, lo que refuerza una idea de examen:
  - el mismo algoritmo matematico puede tener rendimientos muy distintos segun como se alinee con warps y control de flujo.
- Aun asi, la clase advierte que esta version todavia tiene un problema:
  - **conflictos de bancos** en memoria compartida.

### Tercer enfoque de reduccion

- La tercera version usa un esquema descendente:
  - empieza con `i = blockDim.x / 2`;
  - en cada etapa combina `threadIdx.x` con `threadIdx.x + i`;
  - luego divide `i` por dos.
- Conceptualmente, esta reorganizacion busca dos cosas:
  - mantener un patron mas regular de hilos activos;
  - reducir conflictos de bancos en shared memory.
- El problema que resuelve:
  - mejora simultaneamente la estructura de acceso y la ejecucion colectiva del warp.
- La presentacion reporta otra mejora de tiempo, reforzando una leccion metodologica:
  - optimizar reducciones suele consistir en eliminar, una por una, las fuentes de ineficiencia estructural.

### Que enseña la secuencia de tres enfoques

- La clase no presenta tres kernels como variantes arbitrarias.
- Lo que muestra es una forma de pensar optimizacion en CUDA:
  1. primero lograr una solucion correcta;
  2. luego identificar el principal cuello de botella;
  3. corregirlo;
  4. volver a medir y pasar al siguiente.
- En reduccion, los cuellos que aparecen son:
  - divergencia;
  - conflictos de bancos;
  - sincronizaciones repetidas;
  - costo de acceso a memoria global.
- Esta secuencia conecta muy directamente con la clase 8:
  - el rendimiento debe entenderse como consecuencia del patron de computo, no solo del numero de hilos.

### Rol de la sincronizacion en reduce

- La reduccion cooperativa exige **sincronizacion de bloque** entre etapas.
- Cada fase depende de que la fase anterior haya terminado de escribir sus resultados parciales.
- La presentacion remarca un punto importante:
  - no se puede mover una barrera dentro de un `if` si eso hace que no todos los hilos del bloque la alcancen.
- Idea de examen:
  - una barrera de bloque es una operacion colectiva;
  - usarla solo en un subconjunto de hilos rompe la semantica del programa.
- Esto conecta con clases 5 y 7:
  - la sincronizacion siempre depende del **alcance del grupo cooperante**.

### Trade-offs conceptuales del patron reduce

- Ventajas:
  - permite combinar grandes volúmenes de datos con paralelismo local;
  - explota shared memory para amortizar accesos globales;
  - produce una estructura jerarquica escalable.
- Supuestos:
  - la operacion de combinacion debe poder agruparse en etapas;
  - la cooperacion por bloque debe ser suficiente para producir subtotales utiles.
- Costos:
  - requiere sincronizaciones repetidas;
  - su rendimiento depende mucho de divergencia y acceso a shared memory;
  - para llegar a un unico resultado global hace falta una o mas etapas extra.
- Por eso reduce no es solo "sumar en paralelo":
  - es un patron de **agregacion jerarquica**.

### Otras optimizaciones mencionadas

- La presentacion cierra nombrando optimizaciones adicionales:
  - **loop unrolling**;
  - leer mas de una palabra desde memoria global;
  - ajustar ocupacion.
- No las desarrolla, pero su sentido conceptual es claro:
  - una vez controlados los problemas gruesos de divergencia y conflictos, aparecen optimizaciones mas finas de instrucciones, ancho de banda y uso de recursos.

### Comparacion conceptual entre histograma y reduce

- Ambos patrones agregan informacion, pero lo hacen de manera distinta.
- **Histograma**:
  - muchos elementos actualizan posibles destinos compartidos;
  - el desafio dominante es la **contencion**.
- **Reduce**:
  - los valores se combinan en una estructura organizada por etapas;
  - el desafio dominante es la **cooperacion eficiente** dentro del bloque.
- En terminos de memoria y sincronizacion:
  - histograma lleva a pensar en atomicas y privatizacion;
  - reduce lleva a pensar en shared memory, barreras y reordenamiento de accesos.

### Continuidad con los resumenes anteriores

- Conviene reutilizar como contexto:
  - `Clase5-Programacion_CUDA-resumen.md`, por la base de kernels, bloques y sincronizacion de bloque;
  - `Clase6-Programacion_CUDA_2-resumen.md`, porque explica coalescing, atomicas, shared memory y bank conflicts, que son el nucleo conceptual de esta clase;
  - `Clase7-Programacion_CUDA_3-resumen.md`, porque ayuda a ubicar mejor la cooperacion intra-grupo y las restricciones de sincronizacion;
  - `Clase8-Debugging_y_Profiling-resumen.md`, porque las comparaciones de rendimiento entre enfoques se entienden mejor desde una mirada de profiling.
- Esta clase debe leerse como una transicion:
  - ya no se estudian solo mecanismos de CUDA,
  - sino **formas reutilizables de construir algoritmos sobre esos mecanismos**.

## 4. Conceptos clave para memorizar

- **Patron de computo** = estructura reusable de solucion para una familia de problemas paralelos.
- **Histograma** = patron para clasificar muchos elementos en cubetas y contar frecuencias.
- En histogramas, leer la entrada bien requiere accesos **coalesced**.
- En histogramas, el cuello de botella suele estar en las **actualizaciones concurrentes** sobre las cubetas.
- Una operacion **atomica** hace indivisible una secuencia read-modify-write sobre una direccion.
- Las atomicas garantizan correctitud, pero pueden **serializar** actualizaciones en una misma direccion.
- **Privatization** = crear copias parciales del acumulador para reducir contencion y fusionarlas al final.
- Privatizar intercambia mas memoria y una fase extra de combinacion por menos serializacion.
- **Reduce** = patron para combinar muchos valores en uno o pocos resultados agregados.
- En reduccion por bloques, los datos suelen copiarse primero a **shared memory**.
- Una reduccion eficiente depende de:
  - poca divergencia de warps;
  - pocos conflictos de bancos;
  - sincronizacion correcta entre etapas.
- Una barrera como `__syncthreads()` debe ser alcanzada por todos los hilos del bloque.
- Histograma y reduce son ambos patrones de agregacion, pero con problemas estructurales distintos:
  - contencion en histograma;
  - cooperacion jerarquica en reduce.

## 5. Posibles preguntas teoricas de examen

- Que es un patron de computo en GPGPU y por que resulta util pensar los algoritmos de esa manera.
- Que problema resuelve el patron histograma y por que es dificil implementarlo eficientemente en GPU.
- Por que una particion ingenua por secciones puede producir accesos no coalesced en un histograma.
- Que ventaja conceptual tiene el acceso intercalado al procesar la entrada de un histograma.
- Que es una condicion de carrera en una operacion read-modify-write.
- Por que las atomicas resuelven la correctitud pero no necesariamente el rendimiento.
- Que diferencia conceptual hay entre usar atomicas en memoria global y usar acumulacion local en memoria compartida.
- Que es la privatizacion y en que situaciones conviene aplicarla.
- Que problema resuelve el patron reduce y que supuestos necesita sobre la operacion de combinacion.
- Por que shared memory es central en una reduccion por bloques.
- Como afecta la divergencia de warps al rendimiento de una reduccion.
- Que relacion hay entre el patron de indices usado en una reduccion y los conflictos de bancos.
- Por que `__syncthreads()` no puede colocarse de forma que solo lo ejecute una parte del bloque.
- En que se diferencian conceptualmente histograma y reduce como patrones de agregacion.

## 6. Dudas o ambiguedades detectadas en la presentacion, si las hay

- La clase muestra resultados de tiempo para distintas versiones de reduce, pero no detalla la metodologia experimental mas alla del hardware y la cantidad de iteraciones.
- Se mencionan atomicas en cache L2 como mejora respecto de RAM, pero no se desarrolla en que arquitecturas o condiciones practicas esa diferencia se observa con mas claridad.
- La privatizacion se presenta de forma conceptual y esquematica; no se explicita en detalle que granularidad de copia conviene usar en cada caso, por ejemplo por hilo, warp o bloque.
- Las ultimas optimizaciones de reduce se nombran solo de pasada y no se explican sus costos o supuestos.

## Resumenes previos que conviene reutilizar como contexto

- `resumenes/Clase5-Programacion_CUDA-resumen.md`
- `resumenes/Clase6-Programacion_CUDA_2-resumen.md`
- `resumenes/Clase7-Programacion_CUDA_3-resumen.md`
- `resumenes/Clase8-Debugging_y_Profiling-resumen.md`

## Nombre de archivo sugerido

- `resumenes/Clase9-Patrones_de_computo-resumen.md`

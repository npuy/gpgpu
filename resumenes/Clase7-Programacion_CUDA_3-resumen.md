# Resumen Clase 7 - Programacion CUDA III

## 1. Titulo del resumen

**Clase 7 - Programacion CUDA III**

## 2. Temas principales

- Paralelismo dinamico como extension del modelo tradicional de lanzamiento de kernels.
- Warp shuffle como mecanismo de comunicacion intra-warp basado en registros.
- Planificacion independiente de hilos desde Volta y sus consecuencias sobre las suposiciones clasicas a nivel de warp.
- Revision de warp shuffle con sincronizacion explicita y mascaras de participacion.
- Thread Block Clusters como nuevo nivel intermedio de cooperacion entre bloques.
- Restricciones de correctitud asociadas a comunicacion, sincronizacion y memoria compartida distribuida.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase funciona como **tercera parte del bloque de programacion CUDA**.
- La clase 5 habia presentado el modelo base:
  - host y device;
  - kernels;
  - jerarquia `grid -> blocks -> threads`;
  - cooperacion dentro del bloque;
  - separacion entre espacios de memoria.
- La clase 6 habia profundizado en performance y correctitud dentro de ese modelo:
  - coalescing;
  - shared memory;
  - bank conflicts;
  - tiling;
  - manejo de errores.
- Esta clase da un paso distinto:
  - muestra **extensiones del modelo CUDA** que flexibilizan como se genera trabajo, como se comunican los hilos y hasta donde puede llegar la cooperacion.
- Idea global:
  - CUDA ya no se estudia solo como `host lanza kernel` y `bloques independientes`;
  - aparecen mecanismos para **generar trabajo desde la GPU**, **comunicar hilos dentro del warp sin shared memory** y **permitir cierta cooperacion entre bloques**, pero siempre bajo restricciones nuevas.

### Que conceptos previos se profundizan

- **Jerarquia de ejecucion**:
  - antes: `grid -> blocks -> threads`;
  - ahora: se extiende con lanzamientos desde device y con clústeres de bloques.
- **Cooperacion entre hilos**:
  - antes: la cooperacion natural estaba dentro del bloque usando shared memory y sincronizacion;
  - ahora: se estudia una cooperacion mas fina dentro del warp mediante shuffle y una cooperacion mas amplia entre bloques mediante clusters.
- **Modelo SIMT y warps**:
  - en la clase 4 se explico que el warp era la unidad real de planificacion;
  - aqui se profundiza mostrando que esa intuicion clasica cambia con la planificacion independiente de hilos.
- **Sincronizacion**:
  - antes aparecia sobre todo como barrera de bloque;
  - ahora aparece tambien la necesidad de sincronizacion explicita a nivel de warp y de cluster.
- **Memoria compartida**:
  - en la clase 6 era memoria rapida por bloque;
  - aqui se agrega la idea de **memoria compartida distribuida** entre bloques de un mismo cluster.

### Que conceptos son realmente nuevos en esta clase

- **Paralelismo dinamico**.
- **Warp shuffle** y sus variantes.
- **Planificacion independiente de hilos**.
- **Versiones `_sync` de las operaciones shuffle**.
- **Thread Block Clusters**.
- **Memoria compartida distribuida**.

### Paralelismo dinamico

- Antes de **Kepler**, los kernels solo podian ser lanzados desde el **host**.
- El **paralelismo dinamico** agrega la posibilidad de que un kernel lance otro kernel desde el **device**.
- Conceptualmente, esto cambia el flujo clasico de la clase 5:
  - antes el host descubria trabajo, lanzaba kernels y refinaba;
  - ahora parte de ese descubrimiento puede hacerse directamente en la GPU.
- La presentacion lo propone como mecanismo adecuado para problemas con:
  - **recursion**;
  - **loops irregulares**;
  - refinamientos donde el trabajo util se detecta en tiempo de ejecucion.
- Idea importante:
  - el algoritmo puede **concentrar trabajo dinamicamente** donde encuentra mas necesidad de computo.
- Esto profundiza una limitacion previa del modelo CUDA:
  - el esquema tradicional funciona muy bien cuando la estructura del trabajo se conoce de antemano;
  - cuando el trabajo es irregular, depender siempre del host vuelve el refinamiento mas engorroso.

### Restricciones conceptuales del paralelismo dinamico

- Hay **coherencia en memoria global** entre grillas padre e hijas.
- No se puede compartir entre padre e hijo:
  - **memoria local**;
  - **registros**;
  - **memoria compartida**.
- La consecuencia conceptual es clara:
  - el lanzamiento dinamico extiende la capacidad de generar trabajo,
  - pero **no elimina los limites de alcance de memoria** ya estudiados en clases anteriores.
- El material sugiere ademas que la ejecucion del padre se suspende mientras progresa el hijo, aunque no desarrolla el detalle arquitectonico.
- La propia presentacion aclara algo importante para examen:
  - **no es una tecnica de uso masivo en la practica**;
  - sirve mas bien en aplicaciones especificas donde la irregularidad del problema justifica el costo y la complejidad.

### Warp shuffle

- En muchos algoritmos CUDA hace falta comunicar valores entre hilos cercanos.
- La via clasica, trabajada en la clase 6, era usar **shared memory** mas sincronizacion.
- Desde **Kepler**, los hilos de un mismo warp pueden leer valores que otros hilos del warp tienen en sus **registros**.
- Eso se logra con las instrucciones **shuffle**.
- Conceptualmente, shuffle introduce una idea nueva:
  - para la comunicacion **intra-warp**, ya no siempre hace falta pasar por memoria compartida.
- Esto profundiza el estudio del warp de la clase 4:
  - si el warp ya era la unidad natural de ejecucion,
  - ahora tambien aparece como una unidad natural de intercambio de datos.

### Ventaja conceptual de shuffle frente a shared memory

- La presentacion compara ambos enfoques de forma directa.
- Usar shared memory para comunicar datos dentro del warp implica, conceptualmente:
  - escribir;
  - sincronizar;
  - volver a leer.
- En cambio, shuffle resuelve el intercambio con una sola operacion.
- Por eso, a nivel conceptual:
  - **reduce sobrecarga de comunicacion**;
  - **reduce necesidad de shared memory**;
  - puede ayudar cuando la memoria compartida disponible es un recurso escaso.
- La idea importante no es que shuffle reemplace siempre a shared memory:
  - sirve cuando la comunicacion esta confinada al warp;
  - fuera de ese alcance, shared memory o mecanismos mayores siguen siendo necesarios.

### Operaciones shuffle y logica de lanes

- La clase usa el termino **lane** para referirse a la posicion de cada hilo dentro del warp.
- Las operaciones presentadas permiten distintos patrones de intercambio:
  - leer el valor de un lane especifico;
  - copiar desde lanes con desplazamiento hacia arriba o hacia abajo;
  - emparejar lanes segun una mascara XOR.
- Conceptualmente, estas variantes no son detalles sintacticos:
  - definen **patrones de comunicacion intra-warp** reutilizables, por ejemplo para reducciones o broadcasts.
- Tambien aparece el parametro `width`:
  - permite subdividir el warp en subsecciones;
  - cada subseccion se comporta como un grupo independiente para el intercambio.
- Idea conceptual:
  - no todo intercambio intra-warp debe involucrar a los 32 hilos;
  - CUDA permite estructurar cooperacion a granularidad menor.

### Ejemplos de reduccion y continuidad con la clase 6

- La presentacion usa reducciones como ejemplo de uso de shuffle.
- Esto conecta directamente con la clase 6:
  - antes las reducciones servian para pensar uso de shared memory, sincronizacion y optimizacion;
  - ahora se muestra una forma mas directa de implementar la parte intra-warp.
- La leccion conceptual es:
  - una misma operacion paralela puede tener distintas implementaciones segun el nivel de cooperacion involucrado;
  - cuando el intercambio queda contenido en el warp, shuffle puede ser una alternativa mas natural que shared memory.

### Planificacion independiente de hilos

- Esta es una de las ideas mas importantes y potencialmente mas examinables de la clase.
- En arquitecturas con capacidad de computo menor a **7.0**, el warp usaba:
  - un **program counter** compartido;
  - una **mascara de activacion** para indicar que hilos estaban activos.
- Ese esquema coincide con la intuicion clasica explicada en la clase 4:
  - el warp avanza de manera muy acoplada;
  - la divergencia se maneja suspendiendo temporalmente algunos hilos.
- Desde **Volta**, aparece la **planificacion independiente de hilos**.
- Cada hilo mantiene su propio estado de ejecucion, incluyendo:
  - su program counter;
  - su stack de llamadas.
- Luego un optimizador agrupa hilos activos del mismo warp en unidades SIMT para ejecutar en paralelo los que estan siguiendo el mismo camino.

### Que cambia conceptualmente con esta planificacion

- El rendimiento sigue explotando la ejecucion SIMT.
- Pero el modelo se vuelve mucho mas flexible:
  - los hilos pueden diverger y reconverger con granularidad menor a la del warp.
- Esto significa que el warp sigue existiendo como idea de organizacion y eficiencia,
  - pero deja de ser seguro asumir que sus 32 hilos avanzan siempre exactamente al mismo ritmo.
- Esta clase, entonces, **corrige y refina** una simplificacion util de clases anteriores:
  - era razonable pensar el warp como unidad estrictamente sincronica para entender CUDA clasico;
  - en arquitecturas mas nuevas, esa intuicion ya no alcanza para razonar sobre correctitud.

### Restriccion critica: no confiar en sincronismo implicito del warp

- La presentacion advierte que la planificacion independiente de hilos puede romper codigo viejo.
- Ese codigo dependia del comportamiento implicito de que todos los hilos del warp ejecutaban cada instruccion "a la vez".
- Con granularidad subwarp, esa suposicion deja de ser valida.
- Consecuencia conceptual:
  - cuando la correctitud depende de que ciertos hilos del warp hayan llegado a un punto comun o hayan producido datos antes de ser leidos,
  - hay que **sincronizar explicitamente**.
- La operacion destacada para eso es **`__syncwarp()`**.
- Idea de examen:
  - la optimizacion arquitectonica no solo cambia performance;
  - tambien puede invalidar supuestos de programacion que antes parecian seguros.

### Warp shuffle revisitado

- A partir de este cambio de modelo, las operaciones shuffle se revisan.
- Las nuevas variantes agregan dos elementos:
  - una **sincronizacion** asociada a la operacion;
  - una **mascara** que indica que hilos participan.
- Conceptualmente, esto adapta shuffle al nuevo contexto:
  - ya no basta con asumir que todo el warp esta alineado en la misma instruccion;
  - ahora hay que especificar quienes participan y garantizar que los participantes lleguen correctamente.

### Rol conceptual de la mascara en las operaciones `_sync`

- La mascara indica que lanes del warp participan en la operacion.
- La funcion espera a que los hilos no finalizados incluidos en la mascara alcancen la llamada.
- Esto introduce una restriccion de correctitud muy importante:
  - cada hilo que participa debe tener su bit activado;
  - cada hilo que no participa debe tenerlo desactivado.
- Ademas, solo se pueden leer datos de hilos **activamente participantes**.
- Si se intenta leer desde un hilo inactivo, el valor es **indefinido**.
- Idea conceptual central:
  - las primitivas intra-warp modernas siguen siendo muy eficientes,
  - pero ya no pueden pensarse como magia sincronica; requieren declarar con precision el conjunto participante.

### Cooperative Groups como ayuda de abstraccion

- La presentacion menciona **Cooperative Groups** al advertir sobre cuidado en codigo condicional.
- La idea relevante es que CUDA incorpora abstracciones de software para definir grupos de hilos con reglas de sincronizacion mas expresivas.
- No se desarrolla en detalle, pero sirve para ubicarlo conceptualmente:
  - es una capa que ayuda a trabajar con cooperacion mas rica que la jerarquia basica tradicional.

### Thread Block Clusters

- Los **Thread Block Clusters** agregan un nivel nuevo entre grid y bloque.
- Un cluster es un **grupo de bloques de hilos**.
- Puede organizarse en 1, 2 o 3 dimensiones.
- La presentacion aclara que definir clusters:
  - **no cambia** las dimensiones del grid;
  - **no cambia** los indices usuales de bloque dentro del grid.
- Conceptualmente, esto significa que el cluster no reemplaza la jerarquia previa,
  - sino que **agrega una capa de cooperacion estructurada** entre bloques que siguen perteneciendo al mismo grid.

### Que problema viene a resolver el cluster

- En las clases 4 y 5, los bloques debian asumirse esencialmente independientes.
- Esa independencia era la base de la escalabilidad de CUDA.
- El cluster flexibiliza esa regla:
  - ofrece oportunidades de **sincronizacion y comunicacion entre bloques**,
  - pero dentro de una agrupacion controlada y con garantias especiales.
- La presentacion dice que se garantiza que un cluster ejecute en una misma **GPC**.
- La idea importante no es memorizar la sigla, sino entender la implicancia:
  - el hardware asegura cercania suficiente para habilitar cooperacion mas fuerte entre esos bloques.

### Memoria compartida distribuida

- El punto mas nuevo asociado a clusters es la **memoria compartida distribuida**.
- Los hilos de un bloque pueden acceder a la memoria compartida de otros bloques del mismo cluster.
- Eso extiende de forma clara lo visto en la clase 6:
  - antes la shared memory era estrictamente por bloque;
  - ahora aparece un espacio distribuido a nivel de cluster.
- Conceptualmente:
  - la shared sigue estando asignada fisicamente por bloque;
  - pero el software puede tratar el conjunto como un espacio accesible desde otros bloques del cluster.
- Los accesos permitidos incluyen:
  - lectura;
  - escritura;
  - operaciones atomicas.

### Restricciones nuevas de clusters y memoria distribuida

- Esta ampliacion de cooperacion trae problemas nuevos.
- Para acceder correctamente a memoria compartida distribuida:
  - todos los bloques involucrados deben existir;
  - las operaciones deben completarse antes de que finalice un bloque cuyo espacio compartido va a ser accedido;
  - si un bloque remoto va a leer memoria compartida de otro, esa lectura debe terminar antes de que el bloque proveedor finalice.
- La operacion destacada para esto es **`cluster.sync()`**.
- En terminos conceptuales:
  - cuanto mas se expande el alcance de la cooperacion,
  - mas estrictas se vuelven las condiciones de vida y sincronizacion de los datos compartidos.

### Lectura integradora de la clase

- Esta tercera parte de CUDA no se centra en "mas sintaxis", sino en **como evoluciona el modelo de programacion**.
- Lo que se profundiza:
  - el warp como unidad de ejecucion y comunicacion;
  - la cooperacion entre hilos;
  - la relacion entre sincronizacion y correctitud;
  - el alcance real de cada espacio de memoria.
- Lo que se agrega:
  - lanzamiento de kernels desde device;
  - comunicacion por registros dentro del warp;
  - necesidad de sincronizacion explicita intra-warp en arquitecturas modernas;
  - cooperacion controlada entre bloques por medio de clusters.
- La idea final mas importante es:
  - CUDA gana flexibilidad a medida que evoluciona,
  - pero esa flexibilidad nunca es gratuita: siempre viene acompañada por nuevas restricciones semanticas que hay que respetar para conservar correctitud y rendimiento.

## 4. Conceptos clave para memorizar

- **Paralelismo dinamico** = un kernel puede lanzar otro kernel desde el device.
- El paralelismo dinamico sirve especialmente para trabajo **irregular**, recursivo o refinable en tiempo de ejecucion.
- Entre grillas padre e hija hay coherencia de **memoria global**, pero no se comparten registros, memoria local ni shared memory.
- **Warp shuffle** = intercambio de datos entre hilos de un mismo warp usando registros.
- Shuffle reduce uso de **shared memory** y elimina la secuencia escribir-sincronizar-leer cuando la comunicacion es intra-warp.
- **Lane** = posicion de un hilo dentro del warp.
- El parametro `width` permite subdividir un warp en grupos menores para las operaciones shuffle.
- Antes de Volta, el warp compartia un solo **program counter** y una mascara de activacion.
- Desde Volta, la **planificacion independiente de hilos** permite divergencia y reconvergencia a nivel subwarp.
- Ya no es correcto asumir sincronismo implicito del warp para preservar correctitud.
- **`__syncwarp()`** se usa para sincronizar explicitamente hilos del warp.
- Las operaciones **`*_sync`** agregan sincronizacion y una **mask** de participacion.
- Leer datos de un hilo no participante en shuffle produce un valor **indefinido**.
- **Thread Block Cluster** = grupo de bloques con cooperacion y sincronizacion a nivel de cluster.
- **Memoria compartida distribuida** = memoria shared de varios bloques del cluster accesible entre ellos.
- Ampliar el alcance de cooperacion exige controlar cuidadosamente:
  - existencia simultanea de bloques;
  - sincronizacion;
  - tiempo de vida de los datos compartidos.

## 5. Posibles preguntas teoricas de examen

- Que problema del modelo CUDA tradicional intenta resolver el paralelismo dinamico y en que casos resulta especialmente util.
- Que diferencias conceptuales existen entre lanzar kernels desde el host y lanzarlos desde el device.
- Que memorias mantienen coherencia entre grillas padre e hija y cuales no pueden compartirse.
- Que es warp shuffle y por que puede ser mas eficiente que usar shared memory para comunicacion intra-warp.
- Que rol cumplen los lanes y el parametro `width` en las operaciones shuffle.
- Por que la planificacion independiente de hilos modifica la forma de razonar sobre los warps.
- Que suposiciones de codigo antiguo pueden romperse a partir de Volta.
- Por que `__syncwarp()` pasa a ser importante para la correctitud.
- Que informacion expresa la mascara en las versiones `_sync` de shuffle y que restricciones deben cumplirse.
- Que es un Thread Block Cluster y que agrega respecto del modelo `grid -> blocks -> threads`.
- Que se entiende por memoria compartida distribuida y que riesgos introduce.
- Por que ampliar la cooperacion entre bloques obliga a controlar mejor sincronizacion y tiempo de vida de los datos.

## 6. Dudas o ambiguedades detectadas en la presentacion, si las hay

- La presentacion indica que en paralelismo dinamico el padre pareceria suspenderse mientras progresa el hijo, pero no desarrolla con precision el mecanismo de ejecucion.
- Se menciona Cooperative Groups solo de forma introductoria, sin detallar su semantica ni sus costos.
- En Thread Block Clusters se describe la idea general y las restricciones principales, pero no se profundiza en configuracion, limites practicos ni impacto de rendimiento.

## Resumenes previos que conviene reutilizar como contexto

- `resumenes/Clase4-Arquitectura_de_la_GPU-resumen.md`: para repasar warps, SIMT y jerarquia de memoria.
- `resumenes/Clase5-Programacion_CUDA-resumen.md`: para recordar el modelo base host/device, kernels y cooperacion por bloque.
- `resumenes/Clase6-Programacion_CUDA_2-resumen.md`: para conectar shared memory, reducciones y optimizaciones intra-bloque con shuffle y sincronizacion moderna.

## Nombre de archivo sugerido

- `resumenes/Clase7-Programacion_CUDA_3-resumen.md`

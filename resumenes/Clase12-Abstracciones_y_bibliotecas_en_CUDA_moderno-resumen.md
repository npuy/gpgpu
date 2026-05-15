# Resumen Clase 12 - Abstracciones y bibliotecas en CUDA moderno

## 1. Titulo del resumen

**Clase 12 - Abstracciones y bibliotecas en CUDA moderno**

## 2. Temas principales

- Nuevas abstracciones de CUDA para estructurar cooperacion entre hilos con mas claridad.
- `Cooperative Groups` como mecanismo para crear, particionar y sincronizar grupos de threads.
- `Tensor Cores` como hardware especializado para multiplicacion de bloques densos.
- `Building blocks` o primitivas paralelas reutilizables como base de muchos algoritmos GPU.
- Bibliotecas CUDA de distinto nivel de abstraccion, en particular `Thrust` y `CUB`.
- Ejemplos de composicion de primitivas para resolver reducciones y transformaciones mas complejas.
- `CUDA Graphs` como tecnica para reducir overhead cuando hay muchos kernels cortos con dependencias.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase muestra una evolucion natural del recorrido anterior:
  - las clases 5, 6 y 7 habian explicado como programar kernels, organizar hilos y razonar sobre memoria y sincronizacion;
  - la clase 8 habia mostrado como validar y perfilar esos programas;
  - la clase 11 habia introducido el ecosistema CUDA como conjunto de bibliotecas especializadas.
- Aqui el foco pasa a una idea mas moderna:
  - no todo problema en GPU se resuelve escribiendo kernels manuales desde cero;
  - muchas veces conviene usar **abstracciones mas altas** y **bibliotecas optimizadas** que encapsulan patrones paralelos frecuentes.
- La motivacion general de la presentacion es:
  - simplificar la programacion paralela;
  - reutilizar implementaciones eficientes;
  - expresar algoritmos en terminos de primitivas paralelas conocidas.
- Pero la clase deja una advertencia importante:
  - existe un compromiso entre **generalidad**, **productividad** y **performance**;
  - aunque haya mas abstracciones, el conocimiento del hardware sigue siendo fundamental.

### Continuidad con clases anteriores

- Esta clase no reemplaza el modelo CUDA base, sino que se apoya en el.
- Sigue siendo necesario entender:
  - `grid -> blocks -> threads` de la clase 5;
  - warps, cooperacion intra-warp y sincronizacion de la clase 7;
  - costos de memoria, overheads y medicion de performance de las clases 6 y 8.
- La diferencia es que ahora muchas decisiones pueden expresarse en capas mas altas:
  - grupos cooperativos en vez de razonamiento manual hilo por hilo;
  - primitivas de biblioteca en vez de kernels escritos desde cero;
  - grafos de ejecucion en vez de una secuencia de lanzamientos aislados.

### Cooperative Groups

- `Cooperative Groups` aparece como una abstraccion para crear y manipular **grupos cooperativos de hilos**.
- Conceptualmente, extiende y ordena ideas que ya habian aparecido de forma mas manual:
  - en la clase 5, la cooperacion natural estaba dentro del bloque;
  - en la clase 7 aparecian cooperacion intra-warp y sincronizacion mas fina;
  - aqui eso se vuelve una interfaz explicita y reusable.
- La presentacion menciona varios tipos de grupos:
  - `thread_block`;
  - `coalesced_group`;
  - `grid_group`.
- Tambien muestra formas de particionado:
  - **estatico** con `tiled_partition<N>`;
  - **dinamico segun datos** con `labeled_partition` y `binary_partition`.
- Idea conceptual importante:
  - los grupos no son solo un detalle de API;
  - permiten expresar **quien coopera con quien** dentro de un kernel de forma mas clara que trabajando solo con indices y sincronizaciones manuales.
- Ademas, los grupos ofrecen operaciones colectivas dentro del conjunto:
  - `sync`;
  - `reduce`;
  - `shfl`.
- Esto los vuelve especialmente utiles cuando una operacion natural del algoritmo involucra un subconjunto bien definido de hilos.

### Por que Cooperative Groups importa

- La presentacion vuelve a un ejemplo de reducciones en el warp y marca que ciertos patrones irregulares son complejos de implementar solo con `shuffle`.
- La idea es que `Cooperative Groups` ayuda a:
  - abstraer patrones de cooperacion;
  - reducir complejidad accidental;
  - hacer mas natural el trabajo con grupos parciales o definidos por datos.
- Conexion con la clase 7:
  - alli se habia visto que el codigo a nivel de warp podia volverse delicado por sincronizacion y mascara de participacion;
  - aqui aparece una capa que organiza mejor ese tipo de cooperacion.
- Idea de examen:
  - `Cooperative Groups` no cambia el hardware subyacente;
  - cambia la forma de **expresar** cooperacion y operaciones colectivas de manera mas estructurada.

### Tensor Cores

- La segunda gran abstraccion de la clase son los **Tensor Cores**.
- La motivacion presentada es concreta:
  - la multiplicacion de bloques densos es una operacion extremadamente repetida;
  - actualmente esto es especialmente importante en redes neuronales.
- La presentacion lo plantea como agregado de hardware especializado:
  - se incorpora un componente para acelerar esa operacion.
- La clase remarca que, en muchos de estos escenarios, no se requiere tanta precision como en computo numerico clasico.
- El ejemplo textual que aparece es:
  - en Turing, `FP16 x FP16 -> 32`.
- Idea conceptual importante:
  - el hardware se especializa para acelerar una familia de operaciones muy frecuente y con requisitos numericos particulares;
  - no todo el computo de GPU usa la misma ruta de ejecucion general.

### Programacion de Tensor Cores

- La clase dice que los Tensor Cores pueden usarse de dos formas:
  - programandolos directamente con `wmma`;
  - accediendolos indirectamente a traves de bibliotecas como `cublas`.
- Esto resume muy bien el espiritu de toda la clase:
  - hay una opcion mas cercana al hardware y mas explicita;
  - y otra mas productiva apoyada en una biblioteca.
- La interfaz `WMMA` se presenta a nivel de **warp**.
- Los elementos mencionados son:
  - `wmma::fragment` para definir entradas y salidas;
  - `wmma::load` para cargar datos;
  - `wmma::mma_sync` para realizar la multiplicacion;
  - `wmma::fill_fragment` para inicializar;
  - `wmma::store_matrix_sync` para guardar resultados.
- Conceptualmente, lo importante no es memorizar las funciones, sino entender que:
  - el acceso a esta unidad especializada se organiza como una operacion colectiva a nivel de warp;
  - y que las bibliotecas pueden ocultar gran parte de esa complejidad.

### Building Blocks

- La clase introduce la idea de **building blocks** como funciones generales sobre vectores o conjuntos.
- Son primitivas paralelas reutilizables que aparecen una y otra vez en algoritmos GPU.
- Los ejemplos listados son:
  - copias;
  - transformaciones;
  - reducciones;
  - scan;
  - ordenar;
  - merge;
  - permutaciones;
  - search.
- Idea conceptual central:
  - muchos algoritmos complejos pueden descomponerse en unas pocas primitivas paralelas estandar;
  - por eso conviene aprender a reconocer el patron, no solo a escribir kernels ad hoc.

### Copias y reordenamientos

- Dentro de los building blocks, la presentacion distingue:
  - copias contiguas normales;
  - copias `host-device`;
  - reordenamientos como `gather` y `scatter`.
- La idea de fondo es que mover o reordenar datos tambien es una operacion paralela relevante, no solo el calculo aritmetico.
- Esto conecta con la clase 6:
  - la disposicion y el acceso a memoria siguen siendo decisivos para el rendimiento;
  - muchas primitivas existen precisamente para reorganizar datos de manera util para pasos posteriores.

### Transformaciones

- Las transformaciones se presentan como aplicacion paralela de funciones.
- La clase diferencia:
  - `foreach`, generalmente **in-place** y asociado a funciones sencillas;
  - `transform`, generalmente **out-of-place** y apto para funciones de multiples parametros.
- Idea conceptual:
  - no toda operacion paralela acumula o reordena;
  - una gran categoria de algoritmos simplemente aplica una funcion independiente a muchos elementos.
- Esto conserva la intuicion base de CUDA:
  - mismo patron de computo sobre muchos datos,
  - pero ahora encapsulado en una primitiva reusable.

### Reducciones

- La reduccion vuelve a aparecer como building block central.
- Se define como la acumulacion de los elementos en un unico valor usando una operacion binaria.
- La presentacion menciona ejemplos como:
  - suma;
  - maximo;
  - minimo;
  - otras operaciones.
- Esto conecta directamente con clases anteriores:
  - en la clase 3 las reducciones ya eran un patron central de programacion paralela;
  - en clases 5, 6 y 7 aparecian implementaciones CUDA con shared memory o shuffles;
  - aqui se enfatiza que la reduccion puede tratarse como **primitiva reutilizable**.
- Idea de examen:
  - reconocer una reduccion como patron permite delegar su implementacion a bibliotecas altamente optimizadas.

### Scan

- El `scan` acumula los elementos de un vector manteniendo los resultados parciales.
- La clase diferencia:
  - **exclusivo**, que no incluye el valor actual;
  - **inclusivo**, que si lo incluye.
- Ademas, se debe definir:
  - la operacion asociativa usada;
  - un valor inicial, tipicamente `0` o `1`.
- Idea conceptual fuerte:
  - el scan es mas rico que una reduccion porque conserva informacion intermedia;
  - por eso es base de muchas operaciones de compactacion, indices, offsets y particionamiento.

### Ordenamiento, permutaciones, merge, filtrado y busqueda

- La segunda mitad de los building blocks muestra que las primitivas paralelas no se limitan a algebra simple.
- **Ordenar**:
  - ordena un vector;
  - puede ser `stable` o no;
  - puede ser segmentado.
- **Permutaciones**:
  - reordenan elementos;
  - pueden ser aleatorias, utiles por ejemplo en algoritmos estocasticos;
  - o basadas en particiones, como clasificacion binaria por una condicion.
- **Merge**:
  - une dos vectores ordenados.
- **Filtrado**:
  - copia solo una parte del vector;
  - tipicamente se usa para compactar.
- **Busqueda**:
  - encuentra el indice de un elemento;
  - la presentacion menciona busqueda binaria, iterativa y `mismatch`.
- Idea global:
  - muchas tareas de organizacion de datos tienen versiones paralelas estandarizadas;
  - pensar en estos building blocks ayuda a elevar el nivel de diseño del algoritmo.

### Bibliotecas CUDA como capa de abstraccion

- La clase pasa luego de las primitivas conceptuales a las **bibliotecas** concretas.
- La idea principal es:
  - CUDA incluye bibliotecas muy optimizadas;
  - muchas primitivas comunes ya tienen implementaciones eficientes;
  - suelen rendir muy bien en casos estandar.
- Pero la presentacion tambien subraya una decision de diseño:
  - existen bibliotecas de distinto nivel de abstraccion;
  - el programador debe elegir entre **productividad** y **control**.
- Esta idea continua muy directamente la clase 11:
  - el ecosistema CUDA no solo ofrece bibliotecas numericas de dominio especifico;
  - tambien ofrece bibliotecas mas generales de algoritmos y primitivas paralelas.

### Thrust

- `Thrust` se presenta como una extension de la STL para GPUs.
- La clase destaca que:
  - maneja automaticamente memoria y ejecucion;
  - es facil de usar para quien no programa CUDA todos los dias;
  - tiene muchas primitivas paralelas;
  - mantiene interoperabilidad con el resto de CUDA.
- Conceptualmente, `Thrust` representa una opcion de **alto nivel**.
- Sobre memoria, la presentacion distingue dos estilos:
  - manejo automatico con contenedores;
  - manejo manual mediante wrappers de punteros.
- Tambien remarca que:
  - la liberacion automatica existe;
  - pero puede ser mas eficiente gestionar memoria manualmente.
- Idea importante:
  - `Thrust` mejora productividad y expresividad;
  - pero parte del costo de esa comodidad es perder algo de control fino y, a veces, pagar overhead.

### CUB

- `CUB` se presenta como una biblioteca que surge de necesidades de mantenimiento, portabilidad y escalabilidad alrededor de `Thrust`.
- La clase la ubica como:
  - un poco mas bajo nivel que `Thrust`;
  - especifica para CUDA C++;
  - mas cercana al manejo tradicional de memoria de CUDA.
- En `CUB`:
  - la memoria se maneja como en CUDA, con `cudaMalloc`, `cudaFree` y punteros;
  - esa gestion corre por cuenta del programador.
- A cambio, la presentacion la asocia a implementaciones mas eficientes de algoritmos principales.
- Conceptualmente:
  - `CUB` ocupa un punto intermedio entre escribir kernels manuales y usar una biblioteca muy de alto nivel.

### Granularidades en CUB

- La clase dice que `CUB` se divide en tres granularidades:
  - `warp`;
  - `block`;
  - `device`.
- Esto es importante porque alinea la biblioteca con la jerarquia conceptual de CUDA.
- No todas las primitivas tienen el mismo alcance:
  - algunas se piensan dentro del warp;
  - otras dentro del bloque;
  - otras sobre datos a escala de device.
- Ademas, la presentacion menciona operaciones mas especificas:
  - `discontinuity`;
  - `adjacent difference`;
  - `select`;
  - `run-length encoding`;
  - `SpMV`;
  - y soporte a problemas batch, por ejemplo `segmented sort`.
- Idea de examen:
  - una biblioteca eficiente no solo ofrece algoritmos genericos;
  - tambien incorpora variantes adaptadas a granularidad y patrones comunes de GPU.

### Ejemplo de reducciones

- La presentacion incluye un ejemplo comparando algoritmos de reduccion.
- El objetivo conceptual del ejemplo es mostrar que:
  - una misma operacion puede implementarse manualmente o mediante bibliotecas;
  - y el costo real depende de tamano, overhead y calidad de implementacion.
- Lo que el texto explicita es:
  - cada algoritmo se ejecuto 500 veces;
  - en una Nvidia RTX 3090 Ti;
  - ignorando la primera ejecucion por overhead de bibliotecas;
  - con tamanos medios y grandes de la forma `1024 * 2^N`;
  - `Thrust` no ejecuta para `N = 30`.
- La leccion mas importante no es el numero exacto de tiempo, sino la metodologia:
  - al comparar abstracciones, hay que considerar el costo de inicializacion y no solo el kernel idealizado.

### Personalizacion de operadores

- En el contexto del ejemplo de reduccion, la clase plantea una pregunta relevante:
  - que pasa si se quiere usar otro operador?
- Indica que hay varios ya implementados, como maximo y minimo.
- Tambien deja planteado el caso de operadores no predefinidos.
- La idea conceptual es:
  - las bibliotecas cubren muchos casos comunes;
  - pero no todos los algoritmos encajan exactamente en los operadores ya disponibles.
- Esto vuelve sobre el trade-off central de la clase:
  - cuanto mas alto es el nivel de abstraccion, mas importante es que el problema se parezca al caso estandar previsto por la biblioteca.

### Composicion de primitivas: compresion de matrices

- Uno de los mensajes mas importantes de la clase aparece en el ejemplo de compresion de matrices.
- La presentacion dice explicitamente que las bibliotecas:
  - no solo sirven para paralelizar primitivas aisladas;
  - tambien permiten programar rutinas complejas **combinandolas**.
- El ejemplo parte del almacenamiento de matrices dispersas:
  - una representacion tipo `COO` guarda valores con sus indices de fila o columna;
  - esa idea puede mejorarse comprimiendo un indice, como en `CSR`.
- Aunque las diapositivas son muy visuales, los pasos textuales permiten extraer la idea general:
  - ordenar usando la fila como clave;
  - usar el orden resultante para reordenar los otros vectores;
  - trabajar con un vector de unos para contar elementos por fila;
  - obtener informacion sobre cuantas filas distintas hay.
- Conceptualmente, esto ilustra algo muy valioso:
  - un algoritmo aparentemente especializado puede construirse encadenando building blocks como ordenamiento, permutacion y conteo.
- Conexion con las clases 9 y 10 sobre patrones de computo:
  - el foco ya no esta en un kernel puntual, sino en reconocer la composicion de patrones paralelos.

### CUDA Graphs

- El ultimo tema de la clase son los **CUDA Graphs**.
- La motivacion es clara:
  - cuando una aplicacion usa muchos kernels cortos, el overhead de lanzamiento puede volverse relevante.
- La clase recuerda que:
  - en GPUs modernas puede lanzarse mas de un kernel a la vez usando streams;
  - pero a veces existen dependencias entre esas operaciones.
- La idea de los grafos es **capturar esas dependencias**.
- Beneficios mencionados:
  - reducir overhead de lanzamientos;
  - organizar una secuencia dependiente de operaciones de forma mas eficiente.
- Costo mencionado:
  - generar el grafo la primera vez tiene un costo inicial.
- Idea conceptual fuerte:
  - `CUDA Graphs` conviene cuando existe una estructura de ejecucion repetitiva donde el costo de describir una y otra vez la misma secuencia pasa a ser significativo.
- Esto conecta con la clase 8:
  - no solo importa el tiempo dentro del kernel;
  - tambien importa la orquestacion general y el costo del runtime.

### Captura explicita vs estructura de dependencias

- Las ultimas diapositivas sugieren dos formas de trabajar con grafos:
  - **captura de grafo**;
  - **definicion explicita**.
- Aunque la presentacion no desarrolla el detalle, la distincion conceptual es util:
  - una via parte de una ejecucion existente y la captura;
  - la otra construye la estructura de dependencias de manera declarativa.
- Esto refuerza la idea general de la clase:
  - CUDA moderno no solo agrega nuevas unidades de hardware o bibliotecas;
  - tambien agrega formas mas estructuradas de describir trabajo y dependencias.

### Lectura integradora de la clase

- Si la clase 11 mostraba el ecosistema CUDA como conjunto de bibliotecas especializadas, la clase 12 muestra algo mas amplio:
  - abstracciones de cooperacion;
  - primitivas paralelas reutilizables;
  - bibliotecas generales de algoritmos;
  - hardware especializado accesible por API o por biblioteca;
  - estructuras de lanzamiento mas eficientes.
- La conclusion conceptual mas importante es:
  - CUDA moderno no se reduce a escribir kernels a mano;
  - programar bien en GPU tambien implica saber **cuando abstraer**, **cuando reutilizar** y **cuando bajar de nivel** para recuperar control o performance.

## 4. Conceptos clave para memorizar

- **Trade-off central**: generalidad, productividad y performance no siempre maximizan juntas.
- **Cooperative Groups** = abstraccion para formar grupos de hilos, particionarlos y aplicar operaciones colectivas.
- **`tiled_partition<N>`** = particionado estatico de threads en grupos de tamano fijo.
- **`labeled_partition` / `binary_partition`** = particionado dinamico segun datos o condicion.
- **Tensor Cores** = hardware especializado para multiplicacion de bloques densos, muy relevante en IA.
- **WMMA** = interfaz a nivel de warp para usar Tensor Cores directamente.
- **Building blocks** = primitivas paralelas reutilizables como `reduce`, `scan`, `sort`, `merge`, `gather`, `filter`.
- **Reduccion** = acumular todos los elementos en un valor usando una operacion binaria.
- **Scan inclusivo/exclusivo** = acumulacion con resultados parciales, incluyendo o no el elemento actual.
- **Thrust** = biblioteca de alto nivel estilo STL, orientada a productividad.
- **CUB** = biblioteca mas cercana a CUDA C++, con mayor control y alta eficiencia.
- **Granularidades de CUB** = `warp`, `block`, `device`.
- **CUDA Graphs** = representacion de dependencias entre operaciones para reducir overhead de lanzamientos repetidos.
- Las bibliotecas no solo resuelven primitivas aisladas:
  - tambien permiten construir algoritmos complejos por composicion.
- Aunque existan abstracciones modernas, el conocimiento del hardware sigue siendo necesario para elegir bien.

## 5. Posibles preguntas teoricas de examen

- ¿Que problema intentan resolver las nuevas abstracciones en CUDA moderno?
- ¿Que significa el compromiso entre generalidad, productividad y performance en CUDA?
- ¿Que aporta `Cooperative Groups` respecto al uso manual de sincronizacion e indices?
- ¿Que tipos de grupos cooperativos menciona la presentacion y para que sirven conceptualmente?
- ¿Que diferencia hay entre particionado estatico y dinamico de hilos en `Cooperative Groups`?
- ¿Que son los Tensor Cores y por que aparecen como hardware especializado?
- ¿Por que los Tensor Cores son especialmente relevantes para multiplicaciones densas y redes neuronales?
- ¿Que relacion hay entre `WMMA`, warps y Tensor Cores?
- ¿Que se entiende por `building blocks` en programacion paralela sobre GPU?
- ¿Por que operaciones como `reduce`, `scan`, `sort` o `filter` se consideran primitivas fundamentales?
- ¿Que diferencia conceptual hay entre `foreach` y `transform`?
- ¿Que diferencia hay entre un `scan` inclusivo y uno exclusivo?
- ¿Que ventajas y desventajas relativas presentan `Thrust` y `CUB`?
- ¿Por que `CUB` se divide en granularidades `warp`, `block` y `device`?
- ¿Que enseña el ejemplo de compresion de matrices sobre la composicion de primitivas paralelas?
- ¿En que tipo de escenario resulta util `CUDA Graphs`?
- ¿Por que el overhead de lanzamiento puede volverse importante aunque cada kernel sea corto?

## 6. Dudas o ambigüedades detectadas en la presentacion

- Varias diapositivas de ejemplos de `Cooperative Groups`, `WMMA`, `gather`, `scan`, `sort`, `merge`, `filter` y `search` son mayormente visuales y no desarrollan el razonamiento paso a paso.
- El ejemplo de reducciones aporta metodologia de comparacion, pero el PDF extraido no deja visibles los resultados numericos ni la curva exacta de rendimiento.
- El caso de compresion de matrices ilustra bien la idea de composicion de primitivas, pero varios pasos estan explicados sobre figuras y no quedan completamente detallados en texto.
- En `CUDA Graphs` se mencionan captura y definicion explicita, pero la presentacion no profundiza en criterios de uso, costos comparativos o restricciones.

## Contexto recomendado para continuidad

- Conviene reutilizar como contexto:
  - `resumenes/Clase5-Programacion_CUDA-resumen.md`, por la jerarquia de ejecucion y cooperacion basica.
  - `resumenes/Clase6-Programacion_CUDA_2-resumen.md`, por memoria, performance y costo de accesos.
  - `resumenes/Clase7-Programacion_CUDA_3-resumen.md`, por warp-level programming y cooperacion fina entre hilos.
  - `resumenes/Clase8-Debugging_y_Profiling-resumen.md`, por overheads de lanzamiento y lectura de rendimiento.
  - `resumenes/Clase11-Ecosistema_CUDA-resumen.md`, porque conecta directamente con el uso de bibliotecas CUDA.

## Nombre de archivo sugerido

- `resumenes/Clase12-Abstracciones_y_bibliotecas_en_CUDA_moderno-resumen.md`

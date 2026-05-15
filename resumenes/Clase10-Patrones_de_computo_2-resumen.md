# Resumen Clase 10 - Patrones de computo II

## 1. Titulo del resumen

**Clase 10 - Patrones de computo II**

## 2. Temas principales

- Extension del catalogo de patrones de computo en CUDA con **stencil** y **scan**.
- **Stencil** como patron de vecindad regular sobre grillas, tipico de convoluciones, procesamiento de imagenes y metodos iterativos.
- Reutilizacion de datos en stencil mediante **tiling** y uso de **shared memory**.
- **Padding** como tecnica para alinear mejor filas de matrices 2D con los segmentos de memoria global.
- Uso de **memoria constante y cache** para mascaras de convolucion de solo lectura.
- **Scan** como patron de prefijos acumulados que transforma recurrencias secuenciales en computos paralelos.
- Construccion de scan con fases jerarquicas, sincronizacion intra-bloque y variante basada en arbol binario balanceado.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- Esta clase continua directamente la logica de **Patrones de computo I**.
- Si en la clase 9 aparecian patrones donde el problema dominante era:
  - la **contencion** sobre destinos compartidos, como en histograma;
  - o la **combinacion jerarquica** de muchos valores, como en reduce;
  - en esta clase aparecen dos familias nuevas con otro tipo de estructura:
    - **stencil**, donde domina la **vecindad local y la reutilizacion de datos**;
    - **scan**, donde domina la **dependencia de prefijos** y la necesidad de reorganizar una recurrencia secuencial.
- La idea general es importante para examen:
  - distintos patrones paralelos exigen distintas respuestas en CUDA;
  - no todo cuello de botella se resuelve con atomicas o reducciones;
  - a veces el problema central es reutilizar vecinos, alinear accesos o romper dependencias aparentes.

### Continuidad con patrones anteriores

- Conviene leer esta clase como ampliacion del catalogo ya abierto:
  - **histograma**: muchos productores, pocos destinos compartidos, alta contencion;
  - **reduce**: agregacion jerarquica de muchos valores en uno;
  - **stencil**: cada salida depende de una vecindad fija de entradas;
  - **scan**: cada salida depende del prefijo acumulado hasta esa posicion.
- Comparacion conceptual:
  - **histograma** concentra escrituras;
  - **reduce** concentra combinaciones;
  - **stencil** concentra **lecturas reutilizadas** sobre vecinos comunes;
  - **scan** concentra **dependencias logicas** que parecen secuenciales pero pueden reorganizarse.
- Relacion con CUDA:
  - en histograma preocupa la contencion y las atomicas;
  - en reduce preocupa la cooperacion por bloques, divergencia y shared memory;
  - en stencil preocupa el trafico redundante a memoria global y el manejo de bordes;
  - en scan preocupa la sincronizacion entre etapas y la forma de estructurar sumas parciales.

### Patron 1: stencil

- El patron **stencil** actualiza elementos de una grilla usando un patron fijo de vecinos conocido de antemano.
- La presentacion lo describe como un esquema donde:
  - la grilla suele ser de 2 o 3 dimensiones;
  - en cada paso se actualizan todas las celdas;
  - el nuevo valor de una celda depende de elementos adyacentes segun una plantilla regular.
- Suele aparecer en:
  - ecuaciones diferenciales en derivadas parciales;
  - automatas celulares;
  - metodo de Gauss-Seidel;
  - procesamiento de imagenes;
  - el juego de la vida;
  - convoluciones de audio, imagen y video.
- Idea conceptual fuerte:
  - stencil no es un patron de acumulacion global como histograma o reduce;
  - es un patron de **actualizacion local repetida** sobre una estructura espacial.

### Convolucion como caso representativo de stencil

- La convolucion aparece como ejemplo central del patron.
- Cada salida se calcula como una **suma ponderada** de vecinos de la entrada.
- Los pesos vienen dados por una **mascara**.
- En 1D, esto se usa tipicamente en audio.
- En 2D, es natural para filtros de imagen.
- Lo importante para estudiar no es el codigo puntual, sino la estructura:
  - cada hilo puede producir una salida;
  - para hacerlo necesita varios datos cercanos;
  - salidas adyacentes reutilizan muchas de las mismas entradas.
- Esa ultima propiedad marca la diferencia con otros patrones:
  - en reduce se combinan datos hacia adentro hasta dejar pocos resultados;
  - en stencil cada salida sigue existiendo, pero comparte gran parte de su vecindad con salidas cercanas.

### Manejo de bordes en stencil

- La presentacion remarca que en los bordes hay que hacer correcciones.
- Ejemplos mencionados:
  - agregar ceros;
  - repetir valores del borde.
- Esto importa porque en stencil los hilos cercanos al borde no tienen una vecindad completa.
- Problema conceptual:
  - el patron local es regular en el interior;
  - pero deja de ser perfectamente regular en los limites del dominio.
- Implicancia para CUDA:
  - suele aparecer logica condicional en los bordes;
  - esa logica puede introducir trabajo extra o divergencia localizada.

### El cuello de botella clasico de stencil: trafico redundante

- El punto mas importante de la clase sobre stencil es que **salidas adyacentes usan entradas comunes**.
- La diapositiva lo explicita con un ejemplo: un mismo elemento de entrada participa en varios elementos de salida.
- Si cada hilo leyera por su cuenta toda su vecindad desde memoria global:
  - se repetirian muchas lecturas;
  - el ancho de banda global se desperdiciaria;
  - el kernel tenderia a quedar limitado por memoria.
- A diferencia del histograma:
  - el problema principal no es la escritura concurrente sobre una misma direccion;
  - es la **relectura innecesaria** de datos que ya necesitan otros hilos cercanos.
- Esto conecta directamente con la clase 6:
  - el desafio ya no es solo coalescing;
  - tambien es **reutilizacion efectiva**.

### Tiling en stencil

- La respuesta natural en CUDA es **tiling**.
- La idea es cargar en **shared memory** los elementos de entrada necesarios para que todo un bloque calcule un tile de salida.
- Esto permite:
  - traer datos desde memoria global una vez;
  - reutilizarlos para varias salidas vecinas;
  - reducir trafico global redundante.
- La clase formula el problema en terminos de tamanos:
  - si un bloque calcula `T` elementos de salida;
  - necesita `T + Mask_Width - 1` elementos de entrada en 1D.
- La razon es importante:
  - para producir un tile de salidas no alcanza con cargar solo las posiciones centrales;
  - tambien hacen falta los **halos** o bordes de la vecindad.
- Esta es una diferencia clave frente a reduce:
  - en reduce los bloques cooperan para condensar datos;
  - en stencil cooperan para **compartir vecinos**.

### Dos alternativas de diseno para cargar tiles

- La presentacion enumera dos estrategias de diseno:

#### Alternativa 1: bloque del tamaño del tile de salida

- El bloque de hilos coincide con el tile de salida.
- Todos los hilos participan del calculo de salidas.
- Consecuencia:
  - algunos hilos deben cargar mas de un elemento a shared memory.
- Ventaja conceptual:
  - todos los hilos hacen trabajo util de salida.
- Costo:
  - la fase de carga se vuelve desigual o requiere que ciertos hilos trabajen mas.

#### Alternativa 2: bloque del tamaño del tile de entrada

- El bloque coincide con el tile de entrada necesario.
- Todos los hilos cargan un elemento a shared memory.
- Consecuencia:
  - algunos hilos no participan luego en el calculo de salidas.
- Ventaja conceptual:
  - la fase de carga queda mas simple y regular.
- Costo:
  - parte de los hilos quedan como hilos de halo y no producen salida.

#### Lectura comparativa

- Esta comparacion es muy importante para examen porque muestra un trade-off tipico de CUDA:
  - o se simplifica la carga de datos;
  - o se maximiza la proporcion de hilos que calculan salida;
  - pero no siempre ambas cosas a la vez.
- En otras palabras:
  - stencil obliga a equilibrar **regularidad de carga**, **uso de shared memory** y **eficiencia de hilos activos**.

### Padding en stencil 2D

- La clase menciona **padding** para stencils 2D.
- La idea es completar cada fila de la matriz agregando columnas para que coincida mejor con segmentos de memoria global.
- Esto debe leerse como una optimizacion de layout:
  - no cambia la matematica del stencil;
  - cambia la forma en que los datos quedan acomodados para accederlos mejor.
- Relacion con la clase 6:
  - es una extension de las ideas de alineacion y coalescing;
  - cuando las filas no quedan bien alineadas, agregar padding puede mejorar el comportamiento de memoria.
- Trade-off:
  - se consume mas memoria;
  - se gana un patron de acceso potencialmente mas favorable.

### Memoria constante y cache para la mascara

- La presentacion destaca que la **mascara** de convolucion:
  - es leida por todos los hilos;
  - no se modifica;
  - suele ser accedida en las mismas posiciones por todos los hilos de un warp en el mismo instante.
- Por eso, resulta buena candidata para **memoria constante** y su cache.
- La ventaja conceptual es clara:
  - no hace falta gastar shared memory en almacenar la mascara;
  - se aprovecha una memoria especializada para datos de solo lectura compartidos.
- Esto encaja muy bien con el modelo CUDA:
  - datos pequenos, inmutables y leidos de forma uniforme por el warp suelen beneficiarse del camino de memoria constante.
- Comparacion con otros patrones:
  - en histograma y reduce, shared memory se usa sobre todo para acumulacion o cooperacion;
  - en stencil, shared memory se reserva preferentemente para el tile de entrada, mientras que la mascara puede vivir en un espacio de solo lectura mas adecuado.

### Lectura integradora del patron stencil

- Stencil conviene recordarlo como un patron de:
  - **grilla regular**;
  - **vecindad fija**;
  - **alta reutilizacion espacial**;
  - **tratamiento especial de bordes**;
  - **fuerte dependencia del layout y de la jerarquia de memoria**.
- Problemas tipicos de CUDA asociados a stencil:
  - lecturas redundantes desde memoria global;
  - necesidad de cargar halos ademas del tile central;
  - manejo de bordes con logica condicional;
  - tension entre tamano del tile, uso de shared memory y trabajo util por hilo;
  - conveniencia de usar padding y memoria constante.

### Patron 2: scan

- El segundo patron es **scan**, tambien llamado prefijo acumulado.
- La clase lo presenta como un algoritmo clave porque permite convertir estrategias secuenciales en paralelas.
- Definicion dada:
  - toma un operador binario asociativo y un arreglo `[x0, x1, ..., xn-1]`;
  - devuelve `[x0, (x0 ⊕ x1), ..., (x0 ⊕ x1 ⊕ ... ⊕ xn-1)]`.
- Con suma, el resultado son los prefijos acumulados.
- Idea central:
  - scan no reduce el arreglo a un solo valor, como reduce;
  - produce una salida por elemento, pero cada salida resume todo el prefijo anterior.

### Por que scan es importante

- La clase enfatiza que scan sirve para convertir recurrencias secuenciales del tipo:
  - `out[j] = out[j-1] + f(in[j])`
  - en un esquema con:
    - computo independiente de terminos elementales;
    - seguido de una operacion scan.
- Esta observacion es conceptualmente muy fuerte:
  - scan actua como una pieza intermedia que desbloquea paralelismo donde parecia haber dependencia secuencial inevitable.
- En terminos de patrones:
  - reduce resuelve "combinar todo";
  - scan resuelve "combinar todo lo anterior para cada posicion".

### Enfoque naive y por que no sirve

- La presentacion menciona una version paralela naive:
  - asignar un hilo a cada salida;
  - hacer que cada hilo sume todos los elementos necesarios para su prefijo.
- Esto es muy ineficiente porque:
  - repite demasiado trabajo;
  - cada salida recalcula casi todo el prefijo desde cero.
- Comparacion con stencil:
  - en stencil el problema era releer vecinos comunes;
  - en scan naive el problema es **recomputar prefijos comunes**.
- En ambos casos aparece la misma leccion de fondo:
  - detectar y explotar estructura compartida es indispensable para que el paralelismo sea rentable.

### Scan paralelo por etapas

- La clase propone un algoritmo paralelo donde:
  - los valores se cargan a **shared memory**;
  - se itera `log(n)` veces;
  - el `stride` se va duplicando;
  - hacen falta barreras de sincronizacion antes de leer y antes de escribir.
- Esta descripcion ya deja varias ideas para memorizar:
  - scan paralelo es un proceso **jerarquico**, no una suma lineal directa;
  - el uso de shared memory evita depender continuamente de memoria global;
  - la sincronizacion entre etapas es estructural, no accidental.
- Relacion con reduce:
  - ambos patrones usan shared memory y fases sucesivas;
  - pero reduce colapsa informacion hasta dejar un agregado;
  - scan conserva informacion de cada posicion y por eso requiere una organizacion mas rica.

### Scan basado en arbol binario balanceado

- Para mejorar eficiencia, la presentacion introduce una formulacion conceptual mediante un **arbol binario balanceado**.
- Hay dos etapas:
  - una primera navegacion de las hojas hacia la raiz, construyendo sumas parciales;
  - una segunda navegacion inversa para construir las salidas a partir de esas sumas parciales.
- La raiz contiene la suma total de las hojas.
- Esta estructura muestra por que scan puede paralelizarse:
  - en vez de depender linealmente del elemento anterior;
  - se reorganiza el problema como una combinacion jerarquica de profundidad logaritmica.
- Comparacion importante:
  - esta primera fase se parece a una **reduccion**;
  - la segunda fase agrega la informacion necesaria para recuperar todos los prefijos, no solo el total.
- Idea de examen:
  - scan puede verse como **reduce + propagacion estructurada de prefijos parciales**.

### Rol de la sincronizacion en scan

- La clase remarca el uso de barreras antes de leer y antes de escribir en cada iteracion.
- Esto tiene una razon conceptual clara:
  - cada etapa depende de resultados parciales construidos por la etapa anterior;
  - si algun hilo avanza antes de tiempo, puede leer datos inconsistentes.
- Scan, por tanto, es un patron donde la cooperacion intra-bloque es muy intensa.
- Comparacion:
  - en histograma podia haber muchos hilos independientes salvo al actualizar bins;
  - en stencil los hilos cooperan sobre la carga del tile;
  - en scan casi toda la estructura del algoritmo depende de fases coordinadas.

### Lectura integradora del patron scan

- Scan conviene recordarlo como un patron de:
  - **prefijos acumulados**;
  - **operador asociativo**;
  - **reestructuracion de dependencias secuenciales**;
  - **fases jerarquicas en shared memory**;
  - **sincronizacion repetida dentro del bloque**.
- Problemas tipicos de CUDA asociados a scan:
  - version naive con trabajo redundantemente alto;
  - necesidad de organizar el computo en etapas `log(n)`;
  - dependencia fuerte de barreras correctas;
  - necesidad de reutilizar resultados parciales sin volver a memoria global a cada paso.

### Comparacion general entre los cuatro patrones vistos hasta ahora

- **Histograma**:
  - patron de clasificacion y conteo;
  - cuello de botella: contencion por escrituras compartidas;
  - herramienta clave: atomicas y privatizacion.
- **Reduce**:
  - patron de agregacion global;
  - cuello de botella: divergencia, conflictos de bancos y sincronizacion por etapas;
  - herramienta clave: shared memory y reduccion jerarquica.
- **Stencil**:
  - patron de vecindad regular;
  - cuello de botella: relectura de vecinos y manejo de halos/bordes;
  - herramienta clave: tiling, padding, shared memory y memoria constante.
- **Scan**:
  - patron de prefijos acumulados;
  - cuello de botella: dependencia aparente secuencial y coordinacion entre fases;
  - herramienta clave: reorganizacion arbolada, shared memory y barreras.

### Lectura global para examen

- Esta clase amplia el catalogo de patrones y refuerza una conclusion general:
  - en CUDA el algoritmo debe adaptarse a la arquitectura;
  - el tipo de reutilizacion o interferencia entre hilos define la estrategia correcta.
- Regla sintetica:
  - si muchos hilos escriben en pocos destinos, pensar en **atomicas o privatizacion**;
  - si muchos datos deben condensarse, pensar en **reduce**;
  - si muchas salidas comparten vecinos, pensar en **stencil + tiling**;
  - si el problema parece una recurrencia secuencial sobre prefijos, pensar en **scan**.

### Continuidad con los resumenes anteriores

- Conviene reutilizar como contexto:
  - `Clase5-Programacion_CUDA-resumen.md`, por la base de `grid -> blocks -> threads` y cooperacion por bloque.
  - `Clase6-Programacion_CUDA_2-resumen.md`, por coalescing, shared memory, tiling, alineacion y memoria constante.
  - `Clase8-Debugging_y_Profiling-resumen.md`, porque estos patrones suelen requerir medir si el cuello esta en memoria, sincronizacion o contencion.
  - `Clase9-Patrones_de_computo-resumen.md`, porque es la continuacion directa del catalogo de patrones con histograma y reduce.

## 4. Conceptos clave para memorizar

- **Stencil**: actualizacion de cada celda usando una vecindad fija de la grilla.
- **Convolucion**: caso de stencil donde la salida es una suma ponderada por una mascara.
- **Bordes**: requieren tratamiento especial porque no siempre existe vecindad completa.
- **Tiling en stencil**: cargar a shared memory el tile de entrada y sus halos para reutilizar vecinos.
- **Halo**: datos extra alrededor del tile de salida necesarios para calcularlo.
- **Padding**: agregar columnas para mejorar alineacion y acceso a memoria global en 2D.
- **Memoria constante**: conveniente para mascaras de solo lectura compartidas uniformemente por el warp.
- **Scan**: prefijo acumulado sobre un operador asociativo.
- **Scan vs reduce**: reduce deja uno o pocos agregados; scan deja un resultado acumulado por posicion.
- **Arbol binario balanceado en scan**: reorganiza una recurrencia lineal en un proceso jerarquico paralelo.
- **Sincronizacion en scan**: cada etapa depende de la anterior, por eso las barreras son esenciales.

## 5. Posibles preguntas teoricas de examen

- Que distingue al patron stencil de los patrones histograma y reduce en terminos de acceso a datos y cooperacion entre hilos.
- Por que en stencil las salidas adyacentes justifican usar shared memory.
- Que problema resuelven los halos al hacer tiling de una convolucion.
- Compare las dos alternativas de diseno de tiles presentadas para stencil y explique su trade-off.
- Que significa hacer padding en un stencil 2D y que ventaja busca.
- Por que la mascara de una convolucion es buena candidata para memoria constante.
- Defina scan y explique por que requiere un operador asociativo.
- Por que una implementacion naive de scan es paralela pero ineficiente.
- Como convierte scan una recurrencia secuencial en una estrategia paralela.
- Que relacion conceptual hay entre reduce y la primera fase de un scan basado en arbol.
- Que papel cumplen las barreras de sincronizacion en scan.
- Compare los cuellos de botella principales de histograma, reduce, stencil y scan.

## 6. Dudas o ambiguedades detectadas en la presentacion, si las hay

- Las diapositivas de scan muestran el algoritmo y el arbol de forma principalmente grafica, asi que varios detalles finos de implementacion quedan implicitos y no se explican en texto.
- La presentacion menciona memoria constante y `const __restrict__` para la mascara, pero no desarrolla en detalle cuando el compilador efectivamente aprovecha ese camino de cache.
- En stencil se comparan dos alternativas de carga de tiles, pero no se profundiza en que casos practicos conviene cada una.

## Nombre de archivo propuesto

- `resumenes/Clase10-Patrones_de_computo_2-resumen.md`

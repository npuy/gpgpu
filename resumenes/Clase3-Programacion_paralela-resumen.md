# Resumen Clase 3 - Programacion Paralela

## 1. Titulo del resumen

**Clase 3 - Programacion Paralela**

## 2. Temas principales

- Diferencia entre **computacion paralela** y **programacion paralela**.
- Conceptos basicos de coordinacion entre multiples unidades de computo.
- Modelos mentales para descomponer trabajo y datos.
- Scheduling, distribucion de datos/calculos y balance de carga.
- Variables privadas, identificacion del hilo/proceso y trabajo local.
- Reducciones como patron central de agregacion paralela.
- Problemas de correctitud: race conditions, sincronizacion y acceso concurrente.
- Herramientas conceptuales de control: mutex, exclusiones mutuas y operaciones atomicas.
- Divergencia de caminos de ejecucion entre hilos.

## 3. Resumen desarrollado por secciones

### Computacion paralela vs programacion paralela

- **Computacion paralela** es el campo general que estudia como resolver un problema usando varias unidades de procesamiento en simultaneo.
- En la clase 2, esto aparecia como el marco amplio: taxonomias, modelos de memoria, descomposicion funcional o de dominio, costo de comunicacion y sincronizacion.
- **Programacion paralela** es el paso siguiente: como escribir un programa concreto para que esas unidades trabajen coordinadamente, produzcan el resultado correcto y ademas sean eficientes.
- Distincion clave:
  - computacion paralela = **modelo de ejecucion y organizacion del hardware/problema**;
  - programacion paralela = **tecnicas concretas para expresar, repartir y coordinar el trabajo**.
- Relacion con CUDA:
  - la clase 2 explica por que existe el paralelismo;
  - la clase 3 explica que problemas aparecen cuando un programador intenta usarlo;
  - CUDA sera una realizacion concreta de estas ideas sobre GPU.

### Idea central de la clase

- La presentacion define la programacion paralela como **multiples unidades de computo trabajando en forma coordinada**.
- Esa coordinacion obliga a pensar en:
  - **comunicacion**;
  - **sincronizacion**;
  - **division del trabajo**;
  - **division de datos**;
  - enfoques **hibridos**.
- La diapositiva tambien recuerda que el paralelismo puede darse en:
  - **memoria compartida**;
  - **memoria distribuida**;
  - contextos **homogeneos** o **heterogeneos**.
- En este curso, el caso de interes se parece a:
  - muchas unidades de computo;
  - trabajo coordinado;
  - division de datos;
  - unidades relativamente homogeneas;
  - un paralelismo conceptualmente cercano a memoria compartida.

### Modelo mental correcto para pensar un programa paralelo

- El programador no debe pensar solo en "ejecutar mas rapido", sino en **replicar una misma logica sobre multiples porciones de trabajo**.
- Modelo mental util:
  - existe un trabajo total;
  - ese trabajo se parte en subtareas;
  - cada unidad procesa una parte;
  - luego puede hacer falta recombinar resultados.
- Este modelo conecta directamente con la clase 2:
  - la **descomposicion de dominio** sigue siendo la estrategia dominante;
  - la diferencia es que ahora se estudia como expresarla en codigo y que errores puede introducir.
- Para CUDA esto es fundamental:
  - no basta con saber que la GPU favorece paralelismo de datos;
  - hay que saber **como repartir los datos**, **que variables son compartidas o privadas** y **como evitar interferencias entre hilos**.

### Ejemplos iniciales: del caso secuencial al paralelo

- La clase comienza con un ejemplo secuencial simple: sumar por filas una matriz.
- Luego muestra el mismo problema con varias unidades de procesamiento.
- La idea teorica no es el codigo en si, sino el cambio de perspectiva:
  - en secuencial, una sola unidad recorre todo;
  - en paralelo, varias unidades ejecutan partes del mismo patron de calculo.
- Esto refuerza una idea importante:
  - **programar en paralelo no siempre significa inventar otro algoritmo**;
  - muchas veces significa **redistribuir iteraciones o datos** entre varias unidades.

### Descomposicion del trabajo y distribucion de datos

- Una parte central de la clase es como asignar iteraciones o bloques de datos a las unidades disponibles.
- Aparecen dos estrategias de distribucion:
  - **por bloques**;
  - **ciclica**.
- **Distribucion por bloques**:
  - cada unidad recibe una region contigua de trabajo.
  - suele ser simple y puede favorecer localidad.
- **Distribucion ciclica**:
  - el trabajo se reparte alternando elementos o pequenos bloques entre unidades.
  - sirve especialmente cuando el costo por iteracion no es uniforme.
- La clase muestra por que esto importa con el ejemplo de una matriz triangular superior:
  - las primeras filas tienen mas trabajo que las ultimas;
  - si se reparte por bloques grandes, algunas unidades quedan mucho mas cargadas que otras;
  - una distribucion mas fina o ciclica mejora el **balance de carga**.
- Idea de examen:
  - una particion correcta puede seguir siendo mala si reparte trabajo de forma desbalanceada.
- Conexion con clase 2:
  - reaparece explicitamente el criterio de **balance de carga**;
  - tambien reaparece que el costo computacional no siempre coincide con el tamano de los datos.

### Scheduling y balance de carga

- El **scheduling** es la estrategia con la que se asigna trabajo a las unidades de procesamiento.
- La clase no lo formaliza con mucha teoria, pero lo usa como idea transversal:
  - quien hace que parte;
  - en que orden;
  - con que granularidad.
- Un buen scheduling busca:
  - mantener ocupadas a las unidades;
  - evitar que unas terminen mucho antes que otras;
  - reducir trabajo ocioso.
- Esto es especialmente importante en paralelismo masivo:
  - si una parte del trabajo queda mal repartida, el tiempo total termina dominado por la unidad mas lenta o mas cargada.
- Para CUDA esta intuicion luego reaparece en la organizacion de threads, blocks y grids.

### Variables privadas, identidad del hilo y estado local

- La clase introduce el uso de una identificacion local de cada unidad, como `myId()`.
- Eso permite que cada hilo o proceso:
  - conozca quien es;
  - sepa que parte del trabajo le corresponde;
  - acumule resultados parciales propios.
- Tambien aparece el concepto de **variable privada**:
  - cada unidad tiene su propia copia local;
  - esa variable no se comparte directamente con las demas.
- Modelo mental importante:
  - si una variable representa estado intermedio de una sola unidad, conviene que sea privada;
  - si muchas unidades escriben la misma variable compartida, aparecen riesgos de sincronizacion.
- Esta distincion es fundamental para CUDA, donde despues sera crucial separar:
  - datos locales a cada thread;
  - datos compartidos entre varios threads;
  - datos globales visibles por muchos.

### Reducciones: sumar resultados parciales

- La clase presenta varios caminos para calcular la suma total de todos los elementos de una matriz.
- La idea teorica importante es el patron de **reduccion**:
  - cada unidad produce un resultado parcial;
  - luego esos parciales se combinan usando una operacion asociativa, como suma, producto, minimo o maximo.
- La reduccion es un patron central porque permite:
  - explotar paralelismo durante la fase de calculo local;
  - combinar resultados sin que todos escriban continuamente sobre una misma variable global.
- La presentacion incluso destaca operaciones tipicas de reduce:
  - `+`
  - `*`
  - `min`
  - `max`
- Conexion con CUDA:
  - las reducciones son uno de los patrones mas importantes del curso;
  - aparecen en sumas, conteos, histogramas, maximos, minimos y muchas operaciones numericas.
- Idea de examen:
  - una reduccion bien planteada ayuda tanto a **eficiencia** como a **correctitud**, porque evita parte de la contencion sobre un unico dato compartido.

### Race conditions: el problema clasico de la programacion paralela

- La clase remarca uno de los problemas mas importantes de todo el tema: la **race condition**.
- Ocurre cuando varios hilos compiten por acceder y modificar un mismo dato compartido sin suficiente coordinacion.
- Como el orden de ejecucion puede variar, el resultado final tambien puede variar.
- Idea central:
  - en secuencial el orden de las operaciones esta fijado;
  - en paralelo, muchas intercalaciones posibles pueden producir estados distintos.
- La presentacion muestra el caso clasico de dos hilos incrementando una misma variable:
  - ambos leen el mismo valor viejo;
  - ambos calculan el nuevo valor;
  - ambos escriben;
  - una actualizacion se pierde.
- Esto es importante para examen porque condensa el nucleo del problema:
  - **paralelismo no solo introduce complejidad de rendimiento, sino tambien de correctitud**.

### Correctitud vs desempeno

- La ultima diapositiva deja una idea muy importante:
  - no alcanza con lograr buen desempeno;
  - primero el programa debe ser correcto.
- Sin embargo, comprobar correctitud en paralelo es mas dificil que en secuencial porque:
  - hay interacciones entre hilos;
  - el orden real de ejecucion puede cambiar;
  - algunos errores aparecen solo bajo ciertas intercalaciones.
- Esta es una de las lecciones mas importantes para entender CUDA:
  - optimizar una GPU no tiene sentido si el programa produce resultados no deterministas o incorrectos.

### Mecanismos para controlar acceso concurrente

- La clase menciona tres familias de soluciones:
  - **semaforos**;
  - **zonas de exclusion mutua / mutex**;
  - **operaciones atomicas**.

### Exclusiones mutuas y mutex

- Una region mutex permite que solo un hilo entre a una seccion critica a la vez.
- Eso resuelve el problema de correctitud cuando varias unidades quieren modificar el mismo estado compartido.
- Pero introduce un costo fuerte:
  - si solo un hilo entra por vez, el paralelismo efectivo disminuye.
- Idea clave:
  - la sincronizacion protege la correctitud;
  - pero demasiada sincronizacion destruye el beneficio del paralelismo.
- Esta tension entre **seguridad** y **rendimiento** es estructural en programacion paralela.

### Operaciones atomicas

- Una operacion atomica garantiza que una actualizacion sobre cierta variable compartida ocurra como una unidad indivisible.
- Conceptualmente, evita que otros hilos vean un estado intermedio o interfieran en medio de la actualizacion.
- La clase las presenta como alternativa mas fina que encerrar una region mayor dentro de un mutex.
- Ventaja conceptual:
  - protegen una operacion puntual.
- Desventaja conceptual:
  - siguen serializando ese acceso particular y pueden transformarse en un cuello de botella si muchisimos hilos compiten por la misma direccion de memoria.
- Para CUDA, esto es central: las atomicas existen y son utiles, pero deben usarse con criterio.

### Divergencia de caminos de ejecucion

- La clase tambien muestra un ejemplo donde distintos threads siguen caminos distintos segun una condicion.
- Conceptualmente, esto significa que no todos ejecutan exactamente la misma secuencia de instrucciones.
- En esta clase aparece como una observacion general y como una forma de expresar esquemas maestro-esclavo o comportamientos dependientes del identificador del hilo.
- Idea importante para enlazar con CUDA:
  - cuando el paralelismo de datos se rompe porque los hilos toman caminos distintos, la ejecucion puede complicarse y perder eficiencia.
- Aunque la diapositiva no desarrolla el costo arquitectonico en detalle, sirve como introduccion al problema que luego en GPU se conoce como **divergencia**.

### Que deja instalada esta clase para CUDA

- Esta clase fija los problemas base que despues reaparecen con nombres mas concretos en GPU:
  - como dividir el trabajo;
  - como mapear trabajo a unidades de ejecucion;
  - como usar informacion del hilo;
  - como construir resultados globales a partir de resultados parciales;
  - como evitar accesos concurrentes incorrectos;
  - como sincronizar sin destruir el paralelismo.
- En otras palabras:
  - la clase 2 da el vocabulario de **computacion paralela**;
  - la clase 3 da el vocabulario operativo de **programacion paralela**;
  - las clases de CUDA mostraran como todo eso se implementa sobre una GPU real.

## 4. Conceptos clave para memorizar

- **Computacion paralela** = estudio general de como resolver un problema con varias unidades de procesamiento.
- **Programacion paralela** = como escribir programas que expresen, coordinen y controlen ese trabajo paralelo.
- Programar en paralelo exige pensar en:
  - division de trabajo;
  - division de datos;
  - comunicacion;
  - sincronizacion;
  - balance de carga.
- Estrategias de distribucion:
  - por bloques;
  - ciclica.
- Una distribucion correcta no siempre es eficiente; importa mucho el **scheduling**.
- Variable **privada** = cada hilo tiene su propia copia.
- Variable **compartida** = varias unidades pueden leer/escribir el mismo dato.
- **Reduccion** = combinar resultados parciales mediante operaciones como suma, producto, minimo o maximo.
- **Race condition** = resultado dependiente del orden de ejecucion por acceso concurrente no coordinado.
- **Mutex / exclusion mutua** = protegen una seccion critica dejando entrar a un solo hilo por vez.
- **Operacion atomica** = actualizacion indivisible sobre memoria compartida.
- Hay una tension permanente entre **correctitud** y **desempeno**.
- Esta clase es la antesala directa de CUDA porque introduce los problemas practicos que aparecen al lanzar muchos hilos sobre datos compartidos.

## 5. Posibles preguntas teoricas de examen

- Cual es la diferencia entre computacion paralela y programacion paralela.
- Por que no alcanza con conocer la arquitectura paralela para poder programarla bien.
- Que significa que multiples unidades de computo trabajen en forma coordinada.
- Que diferencia hay entre comunicacion, sincronizacion y division del trabajo.
- Que es el balance de carga y por que afecta el rendimiento.
- Que diferencia hay entre distribucion por bloques y distribucion ciclica.
- Por que un problema con trabajo irregular puede requerir una distribucion distinta.
- Que es una variable privada y por que ayuda en programas paralelos.
- Para que sirve conocer el identificador del hilo o proceso.
- Que es una reduccion y por que es un patron tan comun.
- Que es una race condition y por que puede producir resultados distintos entre ejecuciones.
- Que ventajas y desventajas tienen las exclusiones mutuas.
- Que ventajas y limites tienen las operaciones atomicas.
- Por que en programacion paralela correctitud y rendimiento deben analizarse por separado.
- Como prepara esta clase la comprension del modelo de ejecucion de CUDA.

## 6. Dudas o ambiguedades detectadas en la presentacion, si las hay

- La presentacion usa una pseudointaxis inspirada en Matlab/Octave y no un lenguaje paralelo real; sirve para entender ideas, pero algunos detalles exactos de semantica quedan implicitos.
- El termino "paralelismo similar a memoria compartida" aparece de forma orientativa y no formaliza con precision el modelo exacto que luego usara CUDA.
- Los ejemplos de `reduce`, `mutex` y `atomic` son conceptuales; no siempre distinguen con total rigor entre coste logico, coste arquitectonico y semantica concreta de una implementacion real.
- La diapositiva sobre "maestro-esclavo" y caminos distintos por thread introduce la idea, pero no desarrolla en detalle su impacto sobre eficiencia.

## Contexto reutilizable sugerido

- Conviene reutilizar [Clase1-Introduccion-resumen.md](/Users/nicolaspereira/documents/facu/gpgpu/resumenes/Clase1-Introduccion-resumen.md) para mantener continuidad con:
  - `GPU`, `GPGPU`, `CUDA`, `host`, `device`, `kernel`;
  - la idea de **throughput** y de muchas operaciones similares sobre datos distintos.
- Conviene reutilizar [Clase2-Computacion_paralela-resumen.md](/Users/nicolaspereira/documents/facu/gpgpu/resumenes/Clase2-Computacion_paralela-resumen.md) para sostener:
  - memoria compartida vs distribuida;
  - descomposicion funcional vs de dominio;
  - balance de carga, comunicacion y sincronizacion.
- Este resumen deberia reutilizarse especialmente antes de las clases de CUDA, porque instala el vocabulario que despues se traduce a `threads`, `blocks`, sincronizacion, atomicas y reducciones.

## Nombre de archivo sugerido

- `Clase3-Programacion_paralela-resumen.md`

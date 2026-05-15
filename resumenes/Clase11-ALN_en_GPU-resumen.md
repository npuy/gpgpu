# Resumen Clase 11 - ALN en GPUs

## 1. Titulo del resumen

**Clase 11 - ALN en GPUs**

## 2. Temas principales

- Motivacion de usar GPUs para problemas de **algebra lineal numerica (ALN)**.
- Problemas clasicos de ALN: operaciones con matrices y vectores, factorizaciones, autovalores y SVD.
- Distincion entre **ALN densa** y **ALN dispersa**, y como eso cambia algoritmos, bibliotecas y dificultades.
- Diferencia entre **metodos directos** e **iterativos** para resolver sistemas lineales.
- Rol de los estandares y bibliotecas: **BLAS, LAPACK, cuBLAS, cuSOLVER, cuSPARSE, MAGMA, GINKGO**.
- Ideas de implementacion eficiente en GPU: **trabajo a bloques, padding, estrategias hibridas CPU+GPU, multiples GPUs y precision mixta**.
- Importancia de kernels dispersos como **SpMV, SpTRSV y SpGEMM**.
- Conclusiones practicas sobre cuando la GPU aporta mas y lineas de trabajo abiertas.

## 3. Resumen desarrollado por secciones

### Idea central de la clase

- La clase estudia como las GPUs se aplican a problemas de **algebra lineal numerica**, un area central en computacion cientifica.
- La motivacion principal es que muchisimas aplicaciones dependen de resolver operaciones de ALN:
  - optimizacion y simulacion;
  - computacion grafica;
  - control;
  - bases de datos y grafos;
  - redes neuronales.
- La idea global es que la ALN ofrece muchos kernels fundamentales para HPC, y que las GPUs resultan atractivas cuando el problema tiene suficiente trabajo paralelo y suficiente relacion entre computo y transferencia.
- Conexion con los resumenes anteriores:
  - de las **clases 5 a 7** se reutiliza toda la base de CUDA: kernels, jerarquia de ejecucion, memoria y cooperacion;
  - de las **clases 9 y 10** se reutiliza la idea de **patrones de computo**, especialmente en operaciones como multiplicacion matriz-vector, uso de bloques y reutilizacion de datos.

### Que problemas de ALN aparecen

- La clase enumera como problemas clasicos:
  - operaciones con matrices y vectores;
  - multiplicacion matriz-vector y matriz-matriz;
  - factorizaciones **LU, Cholesky y QR**;
  - calculo de valores y vectores propios;
  - descomposicion **SVD**.
- Esto muestra que ALN no es una sola operacion, sino un conjunto grande de problemas con estructuras distintas.
- Idea importante para examen:
  - algunas operaciones son altamente regulares y favorecen mucho a GPU;
  - otras presentan dependencias o accesos irregulares y son mas dificiles de acelerar.

### Tipos de datos y tipos de matrices

- La clase distingue tipos de datos de punto flotante:
  - **simple precision**;
  - **double precision**;
  - numeros complejos en ambas precisiones.
- Tambien distingue tipos de matrices:
  - **densas**;
  - **triangulares**;
  - **de banda**;
  - **dispersas no estructuradas**.
- Esta clasificacion importa porque el costo, el almacenamiento y la estrategia paralela cambian mucho segun el tipo de matriz.
- Idea conceptual fuerte:
  - en ALN el formato de los datos no es secundario; condiciona directamente que biblioteca usar y que tan natural es mapear el problema a GPU.

### Resolucion de sistemas lineales: metodos directos e iterativos

- La clase toma como ejemplo tipico la resolucion de sistemas lineales.
- Presenta dos grandes familias:

#### Metodos directos

- Llegan a la solucion en un numero determinado de pasos.
- En ausencia de errores numericos, obtienen la solucion exacta.
- Un ejemplo central es la **factorizacion LU**.
- Se apoyan en una secuencia de operaciones basicas de ALN.

#### Metodos iterativos

- Construyen una sucesion de aproximaciones.
- Si el metodo converge, se obtiene una solucion aproximada cuyo error satisface cierto criterio.
- El ejemplo mencionado es **Gradiente Conjugado (GC)**.
- Se basan fuertemente en la **multiplicacion matriz-vector**.

#### Comparacion conceptual

- Los metodos directos suelen apoyarse mas en kernels densos y estructurados.
- Los iterativos suelen depender de kernels repetidos, especialmente **SpMV** en el caso disperso.
- Idea de examen:
  - la eleccion entre directo e iterativo no es solo numerica;
  - tambien cambia mucho el perfil computacional y, por lo tanto, la forma de aprovechar la GPU.

### Rol de estandares y bibliotecas en ALN

- La clase enfatiza que, sobre todo en matrices densas, existe un uso intensivo de **estandares y bibliotecas**.
- Aparecen como referencias principales:
  - **BLAS**;
  - **LAPACK**;
  - extensiones multi-core como **SCALAPACK**;
  - en GPU, bibliotecas como **cuBLAS** y **cuSOLVER**.

### BLAS

- **BLAS** resuelve operaciones basicas:
  - vector-vector;
  - matriz-vector;
  - matriz-matriz.
- Soporta diferentes tipos de datos y diferentes tipos de matrices.
- La clase lo ubica como una base historica y conceptual de toda la ALN de alto rendimiento.
- Idea importante:
  - mucho del trabajo en GPU no consiste en inventar algoritmos totalmente nuevos,
  - sino en implementar eficientemente primitivas estandar que despues reutilizan muchas aplicaciones.

### LAPACK

- **LAPACK** ofrece operaciones mas complejas como:
  - factorizacion LU;
  - Cholesky;
  - QR.
- La implementacion de referencia utiliza BLAS como base.
- Esto deja una relacion jerarquica importante:
  - BLAS aporta kernels fundamentales;
  - LAPACK construye algoritmos mas grandes sobre esos kernels.

### ALN dispersa: por que es distinta

- La clase marca una diferencia fuerte entre ALN densa y dispersa.
- En dispersa:
  - hay mucho mas uso de bibliotecas especializadas;
  - los metodos directos pueden destruir la dispersion por **fill-in**;
  - aparecen subproblemas densos dentro de algoritmos dispersos;
  - los metodos iterativos suelen ser mas sencillos de implementar.
- La operacion central pasa a ser **sparse matrix-vector multiplication (SpMV)**.
- Idea conceptual clave:
  - en matrices dispersas el desafio no es solo hacer muchas cuentas;
  - tambien es conservar la estructura dispersa y sobrevivir a accesos irregulares.

### Benchmarks y relevancia en HPC

- La clase menciona tres benchmarks para ubicar la importancia de ALN:
  - **LINPACK**, basado en ALN densa y factorizacion LU, usado para el **Top500**;
  - **HPCG**, basado en operaciones dispersas como SpMV para resolver CG;
  - **graph500**, basado en operaciones vinculables con ALN dispersa y grafos.
- Esto es importante porque muestra que:
  - la ALN no es solo un tema academico;
  - es una base concreta para medir y comparar supercomputadoras.

### Breve historia de ALN en GPU

- La clase repasa trabajos pioneros previos a CUDA y posteriores.
- La idea historica principal es:
  - primero hubo implementaciones muy condicionadas por las limitaciones graficas de la epoca;
  - luego aparecieron mejoras en multiplicacion de matrices, metodos iterativos y factorizaciones;
  - con **CUDA** se consolida la posibilidad de implementar bibliotecas generales como **cuBLAS**.
- Tambien se mencionan mejoras importantes a **gemm**, estrategias **hibridas CPU+GPU** y uso de **padding**.
- La leccion conceptual no es memorizar nombres de papers, sino entender el recorrido:
  - la ALN en GPU evoluciona desde prototipos muy especificos hacia ecosistemas de bibliotecas estables y reutilizables.

### Precision mixta

- La clase dedica varias diapositivas a **precision mixta**.
- La estrategia general es:
  1. resolver o factorizar en una precision mas barata;
  2. refinar el resultado en una precision mayor.
- La motivacion es aprovechar que ciertas precisiones son mucho mas rapidas o baratas, sin resignar tanta exactitud final.
- El material lo muestra con refinamiento iterativo sobre una factorizacion LU.
- Idea importante para examen:
  - la precision mixta busca equilibrar **rendimiento** y **calidad numerica**;
  - no reemplaza a la doble precision en todos los pasos, sino que combina precisiones con roles distintos.
- La clase remarca ademas que el interes por estas tecnicas reaparece con fuerza cuando vuelven a crecer las diferencias de costo entre precisiones, y hoy se extiende incluso a **half precision**.

### ALN densa en GPUs

- La clase presenta la ALN densa como uno de los temas historicamente mas importantes para GPU.
- Razones:
  - esta muy vinculada a computacion grafica;
  - muchos problemas cientificos se apoyan en kernels densos;
  - existe una base fuerte de trabajo sobre **cuBLAS**.
- Idea conceptual:
  - al apoyarse en bibliotecas consolidadas, mejoras en esas bibliotecas impactan en muchas aplicaciones de forma casi automatica.

### Estrategias tipicas para ALN densa

- Aparecen varias ideas recurrentes:
  - **procesamiento por bloques**;
  - **padding**;
  - **estrategias hibridas CPU+GPU**;
  - **concurrencia** entre arquitecturas;
  - uso de **multiples GPUs**.
- El trabajo a bloques importa porque:
  - ordena mejor los accesos;
  - favorece patrones regulares;
  - ayuda a explotar warps y jerarquia de memoria.
- Esta idea conecta directamente con la clase 10:
  - al igual que en stencil y otros patrones, el tiling o bloqueo busca reutilizacion y mejor alineacion con la memoria.

### Estrategias hibridas y multiples GPUs

- La clase insiste en enfoques **CPU+GPU** donde cada etapa corre en la arquitectura mas conveniente.
- Tambien menciona estrategias concurrentes y distribucion entre multiples GPUs.
- Motivos:
  - aumentar poder de computo;
  - disponer de mas memoria;
  - repartir partes distintas del algoritmo.
- Idea de examen:
  - en ALN grande no siempre conviene pensar en GPU aislada;
  - muchas soluciones eficientes son hibridas y aprovechan heterogeneidad.

### Bibliotecas para ALN densa

- La clase nombra:
  - **cuBLAS**;
  - **cuSOLVER**;
  - **MAGMA**;
  - **CULA**;
  - **cuLAPACK**.
- Conceptualmente, estas bibliotecas encapsulan gran parte del conocimiento de optimizacion, por lo que en la practica suelen ser la primera opcion antes de escribir kernels desde cero.

### ALN dispersa en GPUs

- La clase marca que, historicamente, hubo mucho menos trabajo en dispersa que en densa.
- Motivos principales:
  - accesos irregulares;
  - menor regularidad estructural;
  - dificultad mayor para explotar el hardware SIMD/SIMT de la GPU.
- Aun asi, destaca tres grandes familias de operaciones:
  - **SpMV o SpMM**;
  - **SpTRSV**;
  - **SpGEMM**.

### cuSPARSE como biblioteca base

- Se presenta **cuSPARSE** como la biblioteca de NVIDIA para matrices dispersas.
- Soporta distintos tipos de datos:
  - `float`, `double`, `cuComplex`, `cuDoubleComplex`.
- Soporta indices base 0 y base 1.
- Tambien maneja distintos formatos:
  - **COO**;
  - **CSR**;
  - **CSC**;
  - **BSR**;
  - ademas de estructuras densas relacionadas.
- Idea importante:
  - en algebra dispersa el **formato de almacenamiento** es parte central del algoritmo, no solo una decision de serializacion.

### Operaciones soportadas por cuSPARSE

- La clase organiza la biblioteca en tres niveles:
  - vector disperso con vector denso;
  - matriz dispersa con vector denso;
  - matriz dispersa con varios vectores densos o matrices densas.
- Tambien menciona funciones de transformacion de tipos y soporte para resolucion de sistemas triangulares dispersos con dos fases:
  - **analisis simbolico**;
  - **fase numerica**.
- Esto refleja una idea muy importante en dispersa:
  - antes de calcular valores, muchas veces hace falta entender la **estructura** del problema.

### Operacion 1: SpMV y SpMM

- La multiplicacion matriz dispersa-vector aparece como la operacion dominante en muchos metodos iterativos.
- La clase resalta:
  - presencia desde los trabajos pioneros;
  - paralelismo a nivel de filas;
  - abundancia de formatos alternativos para mejorar localidad.
- Conexion con contenidos anteriores:
  - SpMV hereda muchos problemas clasicos de GPU vinculados a memoria global:
  - accesos poco regulares;
  - dificultades para coalescing;
  - distinto trabajo por fila.
- Idea de examen:
  - en ALN dispersa, muchas optimizaciones se centran menos en flops puros y mas en como almacenar y recorrer la estructura dispersa.

### Operacion 2: SpTRSV

- La resolucion de sistemas triangulares dispersos aparece como parte importante de precondicionadores y metodos iterativos.
- La clase menciona dos paradigmas:
  - **por niveles**;
  - **sync-free**.
- El segundo se presenta como mas eficiente y luego adoptado tambien por NVIDIA.
- La razon conceptual es que resolver una triangular implica dependencias, por lo que el problema no es tan naturalmente paralelo como SpMV.
- Entonces, gran parte del trabajo consiste en exponer paralelismo o reducir el costo de la sincronizacion.

### Operacion 3: SpGEMM

- La multiplicacion matriz dispersa-matriz dispersa aparece como una operacion mas dificil.
- La clase remarca que existen menos trabajos y que el problema tiene dos patrones dispersos simultaneos.
- Esto incrementa la complejidad tanto del almacenamiento como de la gestion de accesos y del ensamblado de la salida.
- Conceptualmente, es una operacion relevante hoy por el crecimiento de aplicaciones de grafos.

### Algunos trabajos en FING

- La ultima parte de la clase recorre trabajos de investigacion de FING sobre:
  - factorizacion de matrices generales;
  - inversion de matrices generales;
  - inversion en multiples GPUs;
  - inversion de matrices **SPD**;
  - resolucion de sistemas triangulares;
  - solucion de sistemas dispersos;
  - sistemas tridiagonales, pentadiagonales y heptadiagonales;
  - precision mixta;
  - relacion entre desempeño y consumo energetico;
  - trabajos mas recientes en algebra densa, algebra dispersa y otros aceleradores.
- El valor teorico de esta seccion no esta tanto en memorizar papers, sino en extraer tendencias:
  - uso de estrategias **a bloques**;
  - enfoques **hibridos**;
  - uso de **concurrencia**;
  - extension a **multiples GPUs**;
  - interes por **energia**, **modelado de desempeño** y **compresion**;
  - continuidad del trabajo sobre kernels dispersos eficientes.

### Conclusiones de la clase

- La diapositiva de cierre resume varias ideas generales:
  - las GPUs permiten implementaciones eficientes de muchas operaciones de ALN;
  - cuanto mayor es la relacion entre **computo** y **transferencia**, mayores suelen ser las ganancias;
  - la **precision mixta** puede ayudar;
  - hay amplio potencial para acelerar problemas de computacion cientifica.
- Esta idea sintetiza muy bien el puente con todo el curso:
  - la GPU rinde mejor cuando el patron de computo explota paralelismo masivo y amortiza bien el costo de memoria y transferencias.

### Lineas abiertas

- La clase cierra con direcciones de trabajo futuras:
  - extender enfoques a multiples GPUs;
  - estudiar nuevos problemas y aplicaciones;
  - evaluar en profundidad nuevas arquitecturas como **FPGAs**, **Tensor Cores**, **RT Cores** y otras GPUs;
  - avanzar en configuracion automatica, scheduling dinamico y modelado teorico de desempeño;
  - seguir desarrollando kernels eficientes de algebra dispersa en distintas plataformas.
- Idea importante:
  - ALN en GPU es un campo activo, no un problema cerrado.

## 4. Conceptos clave para memorizar

- **ALN**: algebra lineal numerica aplicada a operaciones fundamentales de computacion cientifica.
- **Metodos directos**: buscan la solucion en un numero finito de pasos; ejemplo, **LU**.
- **Metodos iterativos**: generan aproximaciones sucesivas; ejemplo, **Gradiente Conjugado**.
- **ALN densa**: mas regular, mas madura en bibliotecas y generalmente mas favorable para GPU.
- **ALN dispersa**: dominada por accesos irregulares y formatos de almacenamiento especializados.
- **BLAS**: primitivas basicas de algebra lineal.
- **LAPACK**: algoritmos mas complejos construidos sobre BLAS.
- **cuBLAS / cuSOLVER / cuSPARSE**: bibliotecas clave de NVIDIA para GPU.
- **SpMV**: kernel central en muchos metodos iterativos para matrices dispersas.
- **SpTRSV**: importante en precondicionamiento y mas dificil por dependencias.
- **SpGEMM**: multiplicacion dispersa-dispersa, compleja por manejar dos estructuras dispersas.
- **Fill-in**: perdida parcial de dispersion al aplicar ciertos metodos directos.
- **Precision mixta**: resolver partes en menor precision y refinar en mayor precision.
- **Trabajo a bloques / padding / estrategias hibridas**: tecnicas recurrentes para adaptar ALN al hardware GPU.
- Regla general de rendimiento:
  - **cuanto mayor sea la relacion computo/transferencia, mayor suele ser el beneficio de la GPU**.

## 5. Posibles preguntas teoricas de examen

- ¿Por que la ALN es un area especialmente relevante para GPGPU?
- ¿Que diferencia conceptual hay entre un metodo directo y uno iterativo para resolver sistemas lineales?
- ¿Por que la multiplicacion matriz-vector es tan importante en metodos iterativos?
- ¿Que diferencias principales existen entre ALN densa y ALN dispersa al llevarlas a GPU?
- ¿Por que las matrices dispersas presentan mayores dificultades en GPU que las densas?
- ¿Que rol cumplen BLAS y LAPACK en el ecosistema de ALN?
- ¿Por que cuBLAS y cuSPARSE son importantes desde el punto de vista practico?
- ¿Que es el problema de **fill-in** y por que afecta a los metodos directos en matrices dispersas?
- ¿En que consiste una estrategia de **precision mixta** y que ventaja busca?
- ¿Por que el trabajo a bloques y el padding aparecen repetidamente en ALN densa sobre GPU?
- ¿Que operaciones dispersas destaca la clase y por que cada una es importante?
- ¿Por que SpTRSV es mas dificil de paralelizar que SpMV?
- ¿Que relacion existe entre relacion computo/transferencia y speedup en GPU?
- ¿Por que muchas soluciones eficientes en ALN usan estrategias hibridas CPU+GPU o multiples GPUs?

## 6. Dudas o ambiguedades detectadas en la presentacion

- Varias diapositivas de la seccion **"Algunos trabajos en FING"** parecen apoyarse mucho en figuras, graficas o esquemas visuales y el texto extraido no siempre alcanza para reconstruir todos los detalles metodologicos.
- Las diapositivas 42, 43, 45 y 46 quedaron practicamente sin contenido textual en la extraccion, por lo que puede haberse perdido parte de la explicacion visual de las estrategias de resolucion.
- La presentacion nombra muchas bibliotecas y papers, pero en varios casos no detalla criterios comparativos precisos entre ellas; el foco parece ser panoramico mas que exhaustivo.

## Contexto reutilizable sugerido

- Conviene reutilizar especialmente:
  - `resumenes/Clase5-Programacion_CUDA-resumen.md`
  - `resumenes/Clase6-Programacion_CUDA_2-resumen.md`
  - `resumenes/Clase7-Programacion_CUDA_3-resumen.md`
  - `resumenes/Clase9-Patrones_de_computo-resumen.md`
  - `resumenes/Clase10-Patrones_de_computo_2-resumen.md`
- Motivo:
  - las clases 5 a 7 aportan el modelo CUDA y la jerarquia de memoria;
  - las clases 9 y 10 aportan patrones de acceso, trabajo por bloques y reutilizacion, utiles para entender por que ciertas operaciones de ALN rinden bien o mal en GPU.

## Nombre de archivo sugerido

- `resumenes/Clase11-ALN_en_GPU-resumen.md`

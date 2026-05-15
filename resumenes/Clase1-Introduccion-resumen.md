# Resumen Clase 1 - Introduccion

## 1. Titulo del resumen

**Clase 1 - Introduccion a GPGPU y CUDA**

## 2. Temas principales

- Motivacion para usar GPUs en computacion de proposito general.
- Ubicacion del curso dentro del area de computacion paralela.
- Evolucion historica de las tarjetas graficas.
- Pipeline grafico y su relacion con la evolucion de las GPUs.
- Aparicion de arquitecturas unificadas.
- Tarjetas programables y paso hacia GPGPU.
- CUDA como plataforma de computacion paralela de Nvidia.
- Comparacion conceptual de performance entre CPU y GPU.

## 3. Resumen desarrollado por secciones

### Motivacion

- GPGPU significa **General-Purpose computing on Graphics Processing Units**: usar la GPU, originalmente pensada para graficos, para resolver problemas de proposito general.
- La motivacion principal del area es aprovechar un hardware muy potente y muy difundido para acelerar computos.
- El crecimiento de GPGPU se sostiene en tres factores:
  - fuerte aumento del poder de computo de las GPUs;
  - fuerte aumento del ancho de banda de memoria;
  - arquitectura intrinsecamente paralela, basada en multiprocesadores.
- La industria de videojuegos impulso durante anos la mejora del hardware grafico, y luego la inteligencia artificial reforzo aun mas esa tendencia.
- Las GPUs aparecen en laptops, desktops, celulares, tablets y supercomputadoras, por lo que no son una tecnologia de nicho.
- La presentacion destaca ademas el interes por la eficiencia energetica: no solo importa el rendimiento absoluto, sino la relacion entre desempeno y consumo (GFLOPS/W).
- Los rankings **Top500** y **Green500** se usan como evidencia de la relevancia de las GPUs en computacion de alto desempeno.

### Curso de GPGPU

- El objetivo del curso es introducir el uso de GPUs para resolver problemas de proposito general.
- El foco del curso estara puesto en **CUDA** y en arquitecturas **Nvidia**.
- Otros modelos, como OpenCL, se mencionan solo como referencia general.

### Un poco de historia

- La evolucion del hardware grafico fue desde adaptadores orientados a texto o graficos simples hacia placas capaces de aplicar efectos 2D y 3D cada vez mas complejos.
- En los anos 90 aparecen tarjetas con aceleracion grafica mas sofisticada, lo que lleva a incorporar en hardware partes crecientes del procesamiento grafico.
- Un hito importante es la **GeForce 256** (1999), presentada como la primera GPU de consumo masivo que implementa el pipeline grafico completo.

### El pipeline grafico

- El **pipeline grafico** es la secuencia de etapas que transforma una escena 3D en una imagen 2D mostrada en pantalla.
- La CPU entrega la escena en forma de **primitivas**, usualmente triangulos.
- Etapas mencionadas en la presentacion:
  - **Vertex control**: transforma triangulos segun posicion, orientacion y visibilidad.
  - **VS/T&L (Vertex Shading / Transform and Lighting)**: transforma vertices y les asigna propiedades como color, normales, tangentes y texturas.
  - **Triangle setup**: prepara calculos sobre aristas e interpolaciones.
  - **Raster**: determina que pixeles pertenecen a cada triangulo e interpola valores necesarios para sombrearlos.
  - **Shader**: calcula el color final de cada pixel usando tecnicas como texturas o iluminacion por pixel.
  - **ROP**: aplica operaciones finales, como mezcla de colores, transparencias, antialiasing y descarte de pixeles ocultos.
  - **FBI (Frame Buffer Interface)**: lee y escribe el buffer de imagen.
- Varias de estas etapas son computacionalmente costosas. Esa presion de rendimiento fue una de las razones del crecimiento de la GPU.

### Por que una arquitectura unificada

- En las arquitecturas antiguas, distintas partes del pipeline usaban procesadores especializados distintos, por ejemplo para vertices o para pixeles.
- El problema de ese enfoque es que el hardware puede quedar desbalanceado: una etapa puede saturarse mientras otra queda ociosa.
- La idea de **arquitectura unificada** es usar una sola clase de procesadores programables para distintas etapas del pipeline.
- Esto mejora el aprovechamiento del hardware y prepara el terreno para usar la GPU como dispositivo de computo mas general.

### Tarjetas programables

- Las GPUs antiguas tenian un **pipeline fijo**: las operaciones y su orden estaban predefinidos por el hardware.
- Luego aparecieron los **shaders programables**, primero para vertices y despues para pixeles.
- Entre 1999 y 2006 aumentan mucho las capacidades de programacion:
  - GeForce 3: vertex shaders programables.
  - Radeon 9700: punto flotante de 24 bits.
  - GeForce FX: punto flotante de 32 bits.
  - Xbox 360: arquitectura unificada.
- Al principio, el software era una gran limitacion:
  - se programaba via BIOS, ensamblador especifico o APIs graficas;
  - habia baja portabilidad;
  - las soluciones dependian mucho del modelo exacto de GPU.
- Esto explicaba por que programar GPUs era dificil antes de CUDA.

### CUDA

- **CUDA (Compute Unified Device Architecture)** fue presentada por Nvidia en 2007.
- Segun la presentacion, representa el cambio mas importante en la programacion de proposito general sobre GPUs porque:
  - consolida una arquitectura unificada;
  - cambia radicalmente el software de desarrollo;
  - vuelve mas accesible la GPGPU para desarrolladores no especialistas en graficos.
- Definicion base:
  - **CUDA** es una arquitectura de computacion paralela de Nvidia para resolver problemas de proposito general en GPU.
- CUDA permite programar la GPU mediante extensiones de lenguajes estandar, principalmente **C** y **Fortran**.
- La pila de software incluye:
  - **bibliotecas** de alto nivel, como **CUFFT** y **CUBLAS**;
  - **CUDA Runtime**, que expone APIs para programar;
  - **CUDA Driver**, encargado del control del dispositivo y la transferencia de datos.
- Terminologia base del modelo:
  - **host**: la CPU, que coordina la ejecucion;
  - **device**: la GPU, vista como coprocesador;
  - **kernel**: funcion que se compila para ejecutarse en la GPU;
  - **hilos**: multiples ejecuciones concurrentes del kernel sobre distintos datos.
- Idea central de uso:
  - si una operacion debe ejecutarse muchas veces de forma independiente sobre datos distintos, esa parte puede aislarse como un kernel y ejecutarse masivamente en paralelo en la GPU.

### Comparacion de performance entre CPU y GPU

- La presentacion remarca que CPU y GPU no persiguen exactamente el mismo objetivo de diseno.
- En una **CPU**, gran parte de los transistores se dedica a:
  - prediccion de branches;
  - prefetch de memoria;
  - ejecucion fuera de orden;
  - caches.
- En una **GPU**, una mayor fraccion de transistores se dedica al calculo.
- Esto ayuda a explicar por que la GPU suele ofrecer mayor rendimiento pico en operaciones de punto flotante y mayor ancho de banda de memoria.
- La presentacion muestra formulas para estimar:
  - **pico teorico de FLOPS**;
  - **pico teorico de transferencia de memoria**.
- Tambien senala que no toda comparacion debe hacerse solo por FLOPS: importan precision, consumo energetico y tipo de problema.

### Terminologia base para el resto del curso

- **CPU (Central Processing Unit)**:
  - procesador de proposito general, optimizado para ejecutar una gran variedad de tareas con buen tiempo de respuesta y control complejo.
- **GPU (Graphics Processing Unit)**:
  - procesador originalmente disenado para el pipeline grafico, con gran capacidad de calculo paralelo y alto ancho de banda de memoria.
- **GPGPU**:
  - uso de la GPU para problemas no graficos, aprovechando su paralelismo masivo.
- **Paralelismo de datos**:
  - mismo calculo aplicado simultaneamente sobre muchos elementos distintos de datos.
  - Aunque la presentacion no lo define con ese nombre en una diapositiva formal, es exactamente el modelo que justifica lanzar muchos hilos ejecutando el mismo kernel sobre entradas independientes.
- **Throughput vs latency**:
  - La presentacion no introduce estos terminos de forma explicita, pero la comparacion CPU/GPU permite fijar la distincion.
  - **Latency**: tiempo que tarda una operacion individual en completarse.
  - **Throughput**: cantidad total de trabajo completado por unidad de tiempo.
  - En general, la CPU esta mas orientada a reducir latencia en tareas complejas y con mucho control.
  - La GPU esta mas orientada a maximizar throughput cuando hay muchisimas operaciones similares e independientes.
- **CUDA**:
  - plataforma de Nvidia para explotar ese throughput usando programacion paralela sobre GPU.

## 4. Conceptos clave para memorizar

- GPGPU = uso de GPUs para computacion de proposito general.
- La GPU surge del mundo grafico, pero su arquitectura la vuelve util para calculo masivamente paralelo.
- El pipeline grafico explica historicamente por que las GPUs evolucionaron hacia procesadores cada vez mas potentes y programables.
- La arquitectura unificada elimina la separacion rigida entre procesadores de vertices y de pixeles.
- CUDA masifico la programacion de GPUs al ofrecer un modelo mas accesible y orientado a proposito general.
- CPU y GPU no compiten con el mismo criterio de diseno:
  - CPU: mas control, menor latencia, mas versatilidad.
  - GPU: mas throughput, mas paralelismo, mas ancho de banda.
- El caso ideal para GPU es una operacion repetida muchas veces e independiente entre elementos de datos.
- `host = CPU`, `device = GPU`, `kernel = funcion que corre en GPU`.

## 5. Posibles preguntas teoricas de examen

- Que es GPGPU y por que fue posible su crecimiento en los ultimos anos.
- Cual es la diferencia conceptual entre CPU y GPU.
- Que relacion hay entre el pipeline grafico y el origen de las GPUs modernas.
- Que significa que una GPU tenga arquitectura unificada y por que eso fue importante.
- Por que el pipeline fijo era una limitacion para la programacion general.
- Que es CUDA y por que fue un punto de inflexion en GPGPU.
- Que componentes principales tiene la pila de software de CUDA.
- Que significa ver a la GPU como `device` y a la CPU como `host`.
- En que tipo de problemas tiene sentido usar una GPU.
- Cual es la diferencia entre optimizar para latency y optimizar para throughput.
- Por que una GPU suele tener mayor pico teorico de FLOPS y mayor ancho de banda que una CPU.

## 6. Dudas o ambiguedades detectadas en la presentacion

- Varias diapositivas de motivacion y de arquitectura unificada son principalmente graficas, por lo que muestran tendencias o ejemplos sin siempre formalizar definiciones precisas.
- La distincion **throughput vs latency** no aparece formulada explicitamente con esos terminos; se puede inferir con bastante seguridad a partir de la comparacion arquitectonica entre CPU y GPU.
- La presentacion mezcla ejemplos historicos, comerciales y tecnicos. Para estudiar, conviene priorizar las ideas conceptuales y no memorizar modelos exactos de placas salvo que el docente los use como referencia historica.

## Contexto reutilizable sugerido

- No existe actualmente una carpeta `resumenes` previa en el repo, asi que este resumen funcionaria como base terminologica inicial para los siguientes.
- Cuando se resuman las clases 2, 3 y 4, convendra reutilizar especialmente de este resumen las definiciones de **GPU**, **GPGPU**, **CUDA**, **kernel**, **host/device**, **paralelismo de datos** y **throughput vs latency**.

## Nombre de archivo sugerido

- `Clase1-Introduccion-resumen.md`

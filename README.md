# 🚦 Proyecto 2 – Cruces Inteligentes con Edge AI  

[![Estado](https://img.shields.io/badge/estado-en_desarrollo-blue.svg)]()
[![Plataforma](https://img.shields.io/badge/target-Raspberry%20Pi-green.svg)]()
[![Build](https://img.shields.io/badge/build-Yocto-kirkstone.svg)]()
[![Lenguajes](https://img.shields.io/badge/lenguajes-Python-orange.svg)]()
[![Stack](https://img.shields.io/badge/stack-OpenCV%20%7C%20TensorFlow%20Lite%20%7C%20V4L2%20%7C%20systemd-lightgrey.svg)]()

---


## 1. Introducción  
<p align="justify">
El aumento constante del parque vehicular y la expansión de las ciudades han generado una problemática creciente en la gestión del tránsito y la seguridad vial. En los cruces más concurridos, donde convergen peatones, ciclistas, motocicletas y vehículos particulares, los accidentes y la congestión se presentan con mayor frecuencia debido a la limitada capacidad de los sistemas tradicionales para adaptarse a las condiciones dinámicas del entorno urbano.
</p>
<p align="justify">
En este contexto, la inteligencia artificial embebida (Edge AI) surge como una alternativa tecnológica capaz de ejecutar algoritmos de visión por computador y aprendizaje automático directamente en dispositivos de bajo consumo energético, como el Raspberry Pi. Esta capacidad de procesamiento local permite realizar tareas de detección, clasificación y seguimiento de objetos en tiempo real sin depender completamente de la conectividad a la nube, lo que reduce la latencia y mejora la privacidad de los datos.
</p>
<p align="justify">
El presente proyecto propone el desarrollo de un sistema embebido que funcione como nodo inteligente dentro de una red de monitoreo de tránsito urbano. Cada nodo estará basado en hardware de Raspberry Pi con una cámara periférica y sensores complementarios, ejecutando modelos optimizados de visión artificial con TensorFlow Lite y OpenCV. El objetivo es detectar y clasificar peatones, ciclistas, fauna o vehículos, y con ello ofrecer información útil para la toma de decisiones en la gestión del tránsito y el mejoramiento de la seguridad vial.
</p>
<p align="justify">
El desarrollo de este tipo de sistemas no solo promueve la aplicación práctica de los conocimientos adquiridos en la asignatura de Sistemas Embebidos, sino que también alinea la formación del estudiantado con las tendencias actuales de la industria electrónica, donde convergen la inteligencia artificial, el IoT y la computación de borde.
</p>



## 2. Justificación
<p align="justify">
Las intersecciones urbanas concentran buena parte de las fricciones de la movilidad: trayectorias impredecibles de peatones y ciclistas, picos de congestión y decisiones de conducción tomadas bajo presión. En este entorno cambiante, los esquemas tradicionales de control —basados en temporizaciones fijas o conteos manuales— resultan insuficientes para anticipar comportamientos y reaccionar con la rapidez que exige la seguridad vial. Incorporar inteligencia en el borde (Edge AI) permite llevar el análisis al lugar donde ocurre el fenómeno, reduciendo la latencia, disminuyendo la dependencia de la nube y resguardando la privacidad de quienes transitan.
</p>

<p align="justify">
Este proyecto propone nodos embebidos basados en Raspberry Pi que ejecutan, en tiempo real, modelos livianos de visión por computador para detectar, clasificar y seguir peatones, ciclistas, fauna y vehículos. Al observar el flujo local con granularidad fina (escena a escena), el sistema puede proveer evidencia cuantitativa para ajustar fases semafóricas, activar alertas preventivas o caracterizar riesgos específicos del cruce (por ejemplo, puntos ciegos peatonales en determinadas horas). Así, la solución trasciende el conteo básico y se orienta a decisiones operativas informadas que impactan directamente en la reducción de incidentes y la mejora de la fluidez.
</p>

<p align="justify">
Desde la perspectiva tecnológica, la iniciativa articula un ecosistema embebido moderno: construcción de una imagen de Linux con Yocto Project, integración de OpenCV y TensorFlow Lite, y despliegue en hardware accesible. Esta combinación habilita ciclos de iteración cortos (medición–ajuste–validación) y una escalabilidad pragmática, pues es factible replicar nodos en múltiples cruces con costos razonables y mantenimiento estandarizado.
</p>

<p align="justify">
El proyecto también posee un alto valor formativo, al poner al equipo frente a retos reales de ingeniería: levantamiento y priorización de requerimientos, diseño de arquitecturas de hardware y software, manejo de dependencias, pruebas en campo y validación contra casos de uso. Así, el estudiantado fortalece competencias clave —diseño, integración, validación y documentación— alineadas con las demandas actuales de la industria electrónica y de sistemas embebidos.
</p>


## 4. Descripción y síntesis del problema
El problema que aborda este proyecto no se limita solo al incremento del parque vehicular, sino a la complejidad con que conviven diferentes actores en una intersección urbana. Peatones, ciclistas, automovilistas y hasta fauna se mueven con patrones heterogéneos en espacios reducidos, generando riesgos de colisión e incertidumbre en el flujo de tránsito. Los sistemas actuales —temporizaciones fijas en semáforos y conteos manuales— no son capaces de anticipar comportamientos ni adaptarse a cambios repentinos, por lo que resultan insuficientes para garantizar seguridad y fluidez en estos cruces.

La síntesis de esta problemática señala la necesidad de dotar a los cruces de una visión integral y reactiva. Esto implica desarrollar nodos inteligentes de bajo coste, basados en plataformas como Raspberry Pi, que integren cámaras y sensores para observar la escena local, ejecutar modelos de visión por computador optimizados y generar métricas precisas en tiempo real. Al procesar los datos en el mismo sitio (edge computing), se reducen la latencia y la dependencia de la nube, y se preserva la privacidad de los usuarios. Sin embargo, esta solución exige afrontar desafíos técnicos: adaptar modelos de IA a recursos limitados, integrar hardware heterogéneo y gestionar el sistema operativo mediante herramientas como Yocto Project y TensorFlow Lite.

En esencia, el proyecto plantea diseñar un sistema replicable y escalable capaz de:

⦁ Detectar y clasificar en tiempo real peatones, ciclistas, fauna y vehículos.

⦁ Proveer datos útiles para ajustes semafóricos, activación de alertas preventivas y caracterización de riesgos específicos de cada cruce.

⦁ Facilitar la toma de decisiones operativas y de política pública basadas en evidencias cuantitativas.

Este enfoque no solo responde a una necesidad real del entorno urbano, sino que también representa una oportunidad educativa. La implementación de un nodo embebido con Edge AI obliga a los estudiantes a enfrentarse a retos de selección de hardware, optimización de software y validación de sistemas en condiciones reales, alineándolos con las tendencias de la industria electrónica y de sistemas embebidos.

## 3. Fuentes bibliográficas
<p align="justify">
⦁	Haofeng Wang and Yan Zhang. 2025. Research on pedestrian detection algorithm in dense scenes. In The 4th International Conference on Computer, Artificial Intelligence and Control Engineering (CAICE 2025), January 10–12, 2025, Heifei, China. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3727648.3727655
</p>

<p align="justify">
⦁	Chenxu Li and Chunmei Wang. 2024. Analysis According to the Algorithm of Pedestrian Vehicle Target Detection in Hazy Weather Based on Improved YOLOv8. In 2024 7th International Conference on Artificial Intelligence and Pattern Recognition (AIPR 2024), September 20–22, 2024, Xiamen, China. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3703935.3704068
</p>

<p align="justify">
⦁	Anilcan Bulut, Fatmanur Ozdemir, Yavuz Selim Bostanci, and Mujdat Soyturk. 2023. Performance Evaluation of Recent Object Detection Models
</p>

<p align="justify">
⦁	 W. Kim e I. Jung, “Smart Parking Lot Based on Edge Cluster Computing for Full  Self-Driving Vehicles,” IEEE Access, vol. 10, págs. 115271-115281, 2022. DOI: 10.1109/ACCESS.2022.3208356.
</p>

<p align="justify">
⦁	Komal Saini and Sandeep Sharma. 2025. Smart Road Traffic Monitoring: Unveiling the Synergy of IoT and AI for Enhanced Urban Mobility. ACM Comput. Surv. 57, 11, Article 276 (June 2025), 45 pages. https://doi.org/10.1145/3729217
</p>

<p align="justify">
⦁	Kasra Aminiyeganeh and Rodolfo W. L. Coutinho. 2023. Performance Evaluation of CNN-based Object Detectors on Embedded Devices. In Proceedings of the Int’l ACM Symposium on Design and Analysis of Intelligent Vehicular Networks and Applications (DIVANet ’23), October 30-November-2023, Montreal, QC, Canada. ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/3616392.3623417
</p>

<p align="justify">
⦁	C. -L. Su, W. -C. Lai, Y. -K. Zhang, T. -J. Guo, Y. -J. Hung and H. -C. Chen, "Artificial Intelligence Design on Embedded Board with Edge Computing for Vehicle Applications," 2020 IEEE Third International Conference on Artificial Intelligence and Knowledge Engineering (AIKE), Laguna Hills, CA, USA, 2020, pp. 130-133, doi: 10.1109/AIKE48582.2020.00026. keywords: {Quantization (signal);Convolution;Neural networks;Memory management;Parallel processing;Network architecture;Edge computing;artificial intelligence;edge computing;embedded;vehicle},
</p>

<p align="justify">
⦁	Mohandas, R., Bhattacharya, M., Penica, M., Van Camp, K., & Hayes, M. J. (2020). TensorFlow enabled deep learning model optimization for enhanced realtime person detection using Raspberry Pi operating at the edge. En Proceedings of the Irish Conference on Artificial Intelligence and Cognitive Science (AICS 2020) (Vol. 2771, pp. 157-168). CEUR Workshop Proceedings. Recuperado de https://web.archive.org/web/20220419204828id_/http://ceur-ws.org/Vol-2771/AICS2020_paper_61.pdf
</p>

<p align="justify">
⦁ Oranen, L. (2021). Utilizing deep learning on embedded devices (Master’s thesis, Tampere University, Faculty of Medicine and Health Technology). Tampere University. From https://trepo.tuni.fi/bitstream/handle/10024/133689/OranenLeevi.pdf?sequence=2
</p>

<p align="justify">
⦁ Lightsout. (2015, April 16). Capturing video from two cameras in OpenCV at once [Question on the Stack Overflow forum]. Stack Overflow. Retrieved October 20, 2025, from https://stackoverflow.com/questions/29664399/capturing-video-from-two-cameras-in-opencv-at-once?utm_source=chatgpt.com
</p>

<p align="justify">
⦁ Haris, K.M., Fatima, N.S., Albeez, S.A. (2023). Advanced Vehicle Detection Heads-Up Display with TensorFlow Lite. In: Shakya, S., Balas, V.E., Haoxiang, W. (eds) Proceedings of Third International Conference on Sustainable Expert Systems . Lecture Notes in Networks and Systems, vol 587. Springer, Singapore. https://doi.org/10.1007/978-981-19-7874-6_47
</p>

---
## 📚 Documentación Entrega Preliminar
- [Cronograma del proyecto](docs/CRONOGRAMA.md)
- [Información General del Sistema](docs/INFORMACION-GENERAL.md)
- [Información Técnica del Sistema](docs/INFORMACION-TECNICA.md)

## 📝Bitácoras
- [Bitácora de Kendy](docs/BITACORA-KENDY.md)
- [Bitácora de Elena](docs/BITACORA-ELENA.md)
- [Bitácora de Chris](docs/BITACORA-CHRIS.md)
- [Diagrama de Gantt](https://estudianteccr-my.sharepoint.com/:x:/g/personal/acostchris_estudiantec_cr/ERtFBhxp_XxPtcSzFqkqFTgBtX6mbRqnva7ExeMKVMnOEw?e=FxJHmr)


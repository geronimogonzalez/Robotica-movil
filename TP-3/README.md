# Como utilizar los códgios del TP3
## Entorno de trabajo
### Está configurado para utilizar vscode como entorno de trabajo, abrir desde vscode la carpeta "Ejs" reconoce automaticamente el entorno de trabajo con la carpeta .vscode
### El entorno contiene, todos los script de cada ejercicio del TP (incisos) ; Los archivos de calibración obtenidos en el ejercicio 1 (left.yaml, right.yaml) ; Los ground truth .csv de los dos escenarios utilizados en los ejercicios
### La carpeta .vscode contiene un archivo launch.json que permite ejecutar cada script desde VSCode utilizando la herramienta "Debug with json file". Al seleccionarla se le permite elegir al usuario con que configuración quiere lanzar el archivo actual.
### las configuraciones son:
* una para los items A a F
* DOS para los items G a J, una para cada escenario
* DOS para el nodo ground_truth_publisher, una para cada escenario
### Algunos ejercicios requieren lanzar el script, luego el nodo ground_truth_publisher y luego el bag del dataset, el informe aclara cuales
### Los bag del dataset no están provistos, pueden descargarse de https://docs.openvins.com/gs-datasets.html en el formato bag de ROS2
### Es necesario tener la librería OpenCV instalada (ver documentación de OpenCV, se puede hacer con pip)

# heatmap_yolov5

### 1. Show in screen

#### 1.1 Ubuntu

```xhost +local:docker```

#### 1.2 Windows

Follow the tutorial: [Linux GUI app from Windows hosts](https://medium.com/@potatowagon/how-to-use-gui-apps-in-linux-docker-container-from-windows-host-485d3e1c64a3)

### 2. Build docker-compose

docker-compose up --build -d

### 3. Exec bash in container

docker exec -it heatmap_yolov5_heatmap_1 bash

### 4. Run code

python main.py

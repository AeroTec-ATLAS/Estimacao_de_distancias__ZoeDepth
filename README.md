Este repositório contém um modelo de estimação de distâncias relativas em metros baseado em Zoe depth com os pesos do depth anything.
Existem 4 .py:
  --> calibration-camera: serve para retirar os valores da matriz de calibração da câmara e será utilizado no código principal. É necessário imprimir a imagem calib_img e tirar 15 a 20 fotos com a mesma câmara (existem exemplos na pasta imagens que são da câmara do drone)
  --> depth_to_pointcloud: Este código faz a estimação da distância relativa em metros da pessoa identificada à câmara e o cáculo do número de pixels entre o centro da bbx e o centro do referencial. Este código funciona com um input de imagens e devolve uma nova imagem "colorida" e um ficheiro .txt com os dados recolhidos (há um ex. de output da pasta zip output)
  --> video_local: Este código funciona exatamnete ao anterior mas em vés de usar imagens é live video com a webcam do pc
  --> live_video: Este código ainda se encontra em desenvolvimento, mas também será igual aos 2 anteriores mas o live video vem da câmara do mini drone através de uma porta udp

Link para descarregar um ficheiro que é demasiado pesado para o git gratis :)
https://drive.google.com/file/d/1LRVURtHU89xV5ZeahTgpkclFGVPZ1XQQ/view?usp=drive_link
-----------------------------
        INSTALAÇÕES:
-----------------------------

#  Atualizar o sistema
sudo apt update && sudo apt upgrade -y

# Instalar Python 3, pip, e venv (caso ainda não tenhas)
sudo apt install python3 python3-pip python3-venv -y

# Criar e ativar um ambiente virtual
python3 -m venv zoedepth_env
source zoedepth_env/bin/activate

# Instalar dependências principais
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#  Instalar OpenCV com suporte GStreamer
sudo apt install libopencv-dev python3-opencv -y
pip install opencv-python

# 6. Instalar outras bibliotecas Python necessárias
pip install numpy pillow ultralytics

# Clonar o ZoeDepth
git clone https://github.com/isl-org/ZoeDepth.git
cd ZoeDepth
pip install -r requirements.txt
cd ..
Se houverem poucos erros é "normal" mas deve-se verificar quais são e se necessário confirmar

# Instalar o GStreamer e plugins necessários (para o último código)
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev

---------------------------------------------------------------------------
      CORRER O CODIGO NO TERMINAL DO VS EM LINUX (AMBIENTE VENV)
---------------------------------------------------------------------------      

python3 live_video.py --gst_src "v4l2src device=/dev/video0 ! videoconvert ! appsink" --receiver_ip 127.0.0.1 --dataset nyu --pretrained local::./checkpoints/depth_anything_metric_depth_indoor.pt --yolo_weights yolov8n.pt

--> NOTA IMPORTANTE:
 para esta parte --gst_src "v4l2src device=/dev/video0 ! videoconvert ! appsink" verifica primeiro isto:

ls /dev/video*       --> /dev/video0 em principio deve ser isto, se não for o mesmo numero é só alterar o 0 por 1 no comando

Atera o IP, se não souberes qual é :

sudo apt install nmap

nmap -sn 192.168.1.0/24


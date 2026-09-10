import cv2
from ultralytics import YOLO

# Carrega o modelo YOLO
model = YOLO("yolo11n.pt")

# Caminho do vídeo
video_path = "video.mp4"

# Abre o vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

print("Iniciando detecção de objetos...")
print("Pressione 'q' para sair.")

while True:
    # Lê um frame
    ret, frame = cap.read()

    if not ret:
        print("Fim do vídeo.")
        break

    # Realiza a detecção
    results = model(frame)

    # Obtém o primeiro resultado
    result = results[0]

    # Percorre os objetos detectados
    for box in result.boxes:
        # Coordenadas da bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Confiança da detecção
        confidence = float(box.conf[0])

        # Classe do objeto
        class_id = int(box.cls[0])

        # Nome da classe
        class_name = model.names[class_id]

        # Texto exibido na tela
        label = f"{class_name} {confidence:.2f}"

        # Desenha a bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Exibe o nome da classe
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Mostra o vídeo com as detecções
    cv2.imshow("Deteccao de Objetos - YOLO", frame)

    # Tecla Q encerra o programa
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
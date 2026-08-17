from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt") 
    model.train(
        data="data.yaml",           
        epochs=100,                 
        imgsz=640,                  
        batch=16,                   
        device=0,                   
        workers=4,                  
        project="runs", 
        name="train_final_v2", 
        exist_ok=True               
    )

if __name__ == '__main__':
    train()
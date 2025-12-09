import os
from ultralytics import YOLO
import cv2
import xgboost as xgb
import pandas as pd

def detect_shoplifting(video_path):
    model_yolo = YOLO("C:\\Users\\jashk\\OneDrive\\Desktop\\Suspicious-Activity-Detection-master\\best.pt")
    model = xgb.Booster()
    model.load_model("C:\\Users\\jashk\\OneDrive\\Desktop\\Suspicious-Activity-Detection-master\\model_weights.json")

    cap = cv2.VideoCapture(video_path)

    print('Total Frame', cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('F', 'M', 'P', '4')
    
    # Generate a unique output path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = fr"D:\Deep Learning Projects\Shoplifting Detection\output video\{video_name}_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_tot = 0

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model_yolo(frame, verbose=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot(boxes=False)

            for r in results:
                bound_box = r.boxes.xyxy
                conf = r.boxes.conf.tolist()
                keypoints = r.keypoints.xyn.tolist()

                print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

                for index, box in enumerate(bound_box):
                    if conf[index] > 0.75:
                        x1, y1, x2, y2 = box.tolist()
                        data = {}

                        # Initialize the x and y lists for each possible key
                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]

                        # print(f'Bounding Box {index}: {data}')
                        df = pd.DataFrame(data, index=[0])
                        dmatrix = xgb.DMatrix(df)
                        cut = model.predict(dmatrix)
                        binary_predictions = (cut > 0.5).astype(int)
                        print(f'Prediction: {binary_predictions}')

                        if binary_predictions == 0:
                            conf_text = f'Suspicious ({conf[index]:.2f})'
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 7, 58), 2)
                            cv2.putText(annotated_frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 7, 58), 2)
                        if binary_predictions == 1:
                            conf_text = f'Normal ({conf[index]:.2f})'
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (57, 255, 20), 2)
                            cv2.putText(annotated_frame, conf_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (57, 255, 20), 2)


            cv2.imshow('Frame', annotated_frame)

            out.write(annotated_frame)
            frame_tot += 1
            print('Processed Frame:', frame_tot)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage:
video_path = "C:\\Users\\jashk\\OneDrive\\Desktop\\Suspicious-Activity-Detection-master\\istockphoto-1391833001-640_adpp_is.mp4"
detect_shoplifting(video_path)


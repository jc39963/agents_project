import cv2
import os
import time

def capture_box_only(path_name):
    save_dir = 'data/images'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{path_name}.jpg')
    
    cap = cv2.VideoCapture(0)
    countdown_start = None
    seconds_to_wait = 5

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Primary Portrait Crop (3:4)
        h, w = frame.shape[:2]
        target_w = int((h * 3) / 4)
        start_x = (w - target_w) // 2
        portrait_frame = frame[:, start_x:start_x + target_w]

        # 2. DEFINE THE BOX COORDINATES (The ROI)
        # We define these once so they are used for both drawing and cropping
        box_margin = int(target_w * 0.15)
        top_y, bottom_y = int(h * 0.15), int(h * 0.85) # Tighter vertical crop
        left_x, right_x = box_margin, target_w - box_margin

        # 3. Setup UI
        display_frame = cv2.flip(portrait_frame, 1)
        ui_layer = display_frame.copy()
        
        # Draw the guide (using mirrored coordinates for UI)
        cv2.rectangle(ui_layer, (left_x, top_y), (right_x, bottom_y), (255, 255, 255), 2)

        # 4. Countdown
        if countdown_start is not None:
            elapsed = time.time() - countdown_start
            remaining = seconds_to_wait - int(elapsed)

            if remaining > 0:
                cv2.putText(ui_layer, str(remaining), (target_w//2 - 50, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
            else:
                # Slice the portrait_frame using box coordinates
                boxed_crop = portrait_frame[top_y:bottom_y, left_x:right_x]
                
                success = cv2.imwrite(file_path, boxed_crop)
                if success: print(f"Box Crop Saved: {file_path}")
                break

        cv2.imshow('Box-Only Capture', ui_layer)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and countdown_start is None:
            countdown_start = time.time()
        elif key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_box_only('captured')
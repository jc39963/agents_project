import cv2

# script to capture an image of your clothing?

def capture_clothes(path_name):
    # 0 is usually the default built-in camera
    cap = cv2.VideoCapture(0)

    while True:
        # 'ret' is a boolean (True if it worked), 'frame' is the image array
        ret, frame = cap.read()

        if not ret:
            break

        # This is where perception logic would go (e.g., color math)
        cv2.imshow('Press Space to Capture', frame)

        # Press 'space' to capture the image
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(f'data/images/{path_name}.jpg', frame)
            break


        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_clothes('test')
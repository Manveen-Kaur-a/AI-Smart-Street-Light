import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import sys
import os


MODEL_PATH = "model_unquant.tflite"
LABEL_PATH = "labels.txt"

# None = automatic Arduino detection
ARDUINO_PORT = None

BAUD_RATE = 9600
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.75
STABLE_FRAMES_REQUIRED = 5
INFERENCE_EVERY_N_FRAMES = 3


def print_banner():
    print("""
================================================
       AI SMART STREET LIGHT SYSTEM
       Teachable Machine + Arduino UNO
================================================
""")


def print_status(icon, message):
    print(f"  {icon}  {message}")


def clean_label(label):
    label = label.strip().lower()

    parts = label.split(" ", 1)

    if len(parts) == 2 and parts[0].isdigit():
        label = parts[1]

    return label


def get_light_condition(label):
    label = label.lower()

    if "day" in label:
        return "day"

    if "night" in label:
        return "night"

    return "unknown"


def find_arduino_port():
    ports = serial.tools.list_ports.comports()

    for port in ports:
        desc = (port.description or "").lower()
        mfr = (port.manufacturer or "").lower()

        if any(x in desc for x in [
            "arduino",
            "ch340",
            "cp210",
            "usb-serial",
            "usb serial"
        ]):
            return port.device

        if any(x in mfr for x in [
            "arduino",
            "wch",
            "silicon labs",
            "ftdi"
        ]):
            return port.device

    if ports:
        print_status("!!", "Available serial ports:")

        for port in ports:
            print(
                f"       {port.device}: "
                f"{port.description}"
            )

        print_status(
            ">>",
            "Set ARDUINO_PORT manually in Street.py"
        )

    else:
        print_status(
            "XX",
            "No serial ports found!"
        )

    return None


def draw_confidence_bar(
    frame,
    x,
    y,
    width,
    confidence,
    condition
):
    height = 20

    filled = int(width * confidence)

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (40, 40, 40),
        -1
    )

    if condition == "day":
        color = (0, 200, 255)
    elif condition == "night":
        color = (255, 150, 0)
    else:
        color = (100, 100, 100)

    cv2.rectangle(
        frame,
        (x, y),
        (x + filled, y + height),
        color,
        -1
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        (200, 200, 200),
        1
    )

    cv2.putText(
        frame,
        f"{confidence * 100:.0f}%",
        (x + width + 10, y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


def draw_osd(
    frame,
    condition,
    confidence,
    fps,
    connected
):
    h, w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 95),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.6,
        frame,
        0.4,
        0,
        frame
    )

    cv2.putText(
        frame,
        "AI Smart Street Light",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if condition == "day":
        text = "DAY - Lights OFF"
        color = (0, 200, 255)

    elif condition == "night":
        text = "NIGHT - Lights ON"
        color = (255, 150, 0)

    else:
        text = "DETECTING..."
        color = (100, 100, 100)

    cv2.putText(
        frame,
        text,
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )

    draw_confidence_bar(
        frame,
        15,
        68,
        200,
        confidence,
        condition
    )

    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",
        (w - 120, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    connection_text = (
        "Arduino: Connected"
        if connected
        else "Arduino: DISCONNECTED"
    )

    connection_color = (
        (0, 255, 0)
        if connected
        else (0, 0, 255)
    )

    cv2.putText(
        frame,
        connection_text,
        (w - 220, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        connection_color,
        1
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


def main():

    print_banner()

    # =========================================
    # ARDUINO
    # =========================================

    print_status(">>", "Connecting to Arduino...")

    arduino = None
    arduino_connected = False

    port = ARDUINO_PORT or find_arduino_port()

    if port:
        try:
            arduino = serial.Serial(
                port,
                BAUD_RATE,
                timeout=1
            )

            time.sleep(2)

            # Start with lights OFF
            arduino.write(b'D')

            time.sleep(0.5)

            arduino_connected = True

            print_status(
                "OK",
                f"Arduino connected on {port}"
            )

            if arduino.in_waiting:
                message = (
                    arduino.readline()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                    .strip()
                )

                print_status(
                    ">>",
                    f"Arduino: {message}"
                )

        except Exception as e:
            print_status(
                "!!",
                f"Arduino connection failed: {e}"
            )

    else:
        print_status(
            "!!",
            "Arduino not found."
        )


    # =========================================
    # MODEL
    # =========================================

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = os.path.join(
        script_dir,
        MODEL_PATH
    )

    label_path = os.path.join(
        script_dir,
        LABEL_PATH
    )

    if not os.path.exists(model_path):
        print_status(
            "XX",
            f"Missing: {MODEL_PATH}"
        )
        sys.exit(1)

    if not os.path.exists(label_path):
        print_status(
            "XX",
            f"Missing: {LABEL_PATH}"
        )
        sys.exit(1)

    print_status(
        ">>",
        "Loading AI model..."
    )

    try:
        try:
            from tflite_runtime.interpreter import Interpreter

            interpreter = Interpreter(
                model_path=model_path
            )

            print_status(
                "OK",
                "Loaded using TFLite Runtime"
            )

        except ImportError:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(
                model_path=model_path
            )

            print_status(
                "OK",
                "Loaded using TensorFlow"
            )

        interpreter.allocate_tensors()

        input_details = (
            interpreter.get_input_details()
        )

        output_details = (
            interpreter.get_output_details()
        )

    except Exception as e:
        print_status(
            "XX",
            f"Model error: {e}"
        )
        sys.exit(1)


    # =========================================
    # LABELS
    # =========================================

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        labels = [
            clean_label(line)
            for line in file
            if line.strip()
        ]

    print_status(
        "OK",
        f"Labels: {labels}"
    )


    # =========================================
    # CAMERA
    # =========================================

    print_status(
        ">>",
        "Starting camera..."
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print_status(
            "XX",
            "Camera not found!"
        )
        sys.exit(1)

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    print_status(
        "OK",
        "Camera started"
    )

    print()
    print("System READY!")
    print("Show DAY or NIGHT to camera.")
    print("Press Q to quit.")
    print()


    # =========================================
    # VARIABLES
    # =========================================

    last_prediction = None
    stable_count = 0
    last_command = None

    frame_count = 0

    condition = "unknown"
    confidence = 0.0

    fps = 0.0
    fps_start = time.time()
    fps_frames = 0


    # =========================================
    # MAIN LOOP
    # =========================================

    while True:

        ret, frame = camera.read()

        if not ret:
            continue

        frame_count += 1
        fps_frames += 1

        elapsed = time.time() - fps_start

        if elapsed >= 1:
            fps = fps_frames / elapsed
            fps_frames = 0
            fps_start = time.time()


        # Run AI every N frames
        if frame_count % INFERENCE_EVERY_N_FRAMES == 0:

            image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            image = np.asarray(
                image,
                dtype=np.float32
            )

            image = (
                image / 127.5
            ) - 1.0

            input_data = np.expand_dims(
                image,
                axis=0
            )

            interpreter.set_tensor(
                input_details[0]["index"],
                input_data
            )

            interpreter.invoke()

            prediction = interpreter.get_tensor(
                output_details[0]["index"]
            )

            index = int(
                np.argmax(prediction)
            )

            confidence = float(
                prediction[0][index]
            )

            label = (
                labels[index]
                if index < len(labels)
                else "unknown"
            )

            detected = get_light_condition(
                label
            )

            if confidence < CONFIDENCE_THRESHOLD:
                detected = "unknown"

            condition = detected


            # Stability
            if detected == last_prediction:
                stable_count += 1
            else:
                stable_count = 1
                last_prediction = detected


            # Send command
            if stable_count >= STABLE_FRAMES_REQUIRED:

                command = None

                if detected == "day":
                    command = b"D"

                elif detected == "night":
                    command = b"N"


                if (
                    command is not None
                    and command != last_command
                ):

                    if arduino_connected:

                        try:
                            arduino.write(command)

                            last_command = command

                            if command == b"D":
                                print_status(
                                    ">>",
                                    f"DAY {confidence:.0%}"
                                    " -> Lights OFF"
                                )

                            else:
                                print_status(
                                    ">>",
                                    f"NIGHT {confidence:.0%}"
                                    " -> Lights ON"
                                )

                        except Exception as e:
                            print_status(
                                "!!",
                                f"Arduino error: {e}"
                            )

                            arduino_connected = False

                    else:
                        last_command = command


        # Display
        display = frame.copy()

        draw_osd(
            display,
            condition,
            confidence,
            fps,
            arduino_connected
        )

        cv2.imshow(
            "AI Smart Street Light",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key in [ord("q"), ord("Q")]:
            break


    # =========================================
    # CLEANUP
    # =========================================

    print_status(
        ">>",
        "Shutting down..."
    )

    if arduino_connected and arduino:

        try:
            arduino.write(b"D")
            time.sleep(0.5)
            arduino.close()

            print_status(
                "OK",
                "Arduino disconnected. LEDs OFF."
            )

        except Exception:
            pass

    camera.release()
    cv2.destroyAllWindows()

    print_status(
        "OK",
        "System stopped. Goodbye!"
    )


if __name__ == "__main__":
    main()
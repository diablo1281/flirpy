import cv2
import numpy as np
from time import time
from flirpy.camera.lepton import Lepton


def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)


with Lepton() as camera:
    img_raw = camera.grab(strip_telemetry=False)

    print('Major version: {}'.format(camera.telemetry["major_version"]))
    print('Minor version: {}'.format(camera.telemetry["minor_version"]))
    print('Uptime: {} ms'.format(camera.telemetry["uptime_ms"]))
    print('Status: {}'.format(camera.telemetry["status"]))
    print('Revision: {}'.format(camera.telemetry["software_rev"]))
    print('Frame count: {}'.format(camera.telemetry["frame_counter"]))
    print('Video format: {}'.format(camera.telemetry["video_output_format"]))
    print('TLinear enable: {}'.format(camera.telemetry["TLinear_enable"]))

    try:
        loop_time = time()
        loop_count = 0
        while True:
            if (time() - loop_time) >= 10.0:
                print("Loop freq: {:.4f} Hz [{:.2f} s, {} times]".format(loop_count / (time() - loop_time), time() - loop_time, loop_count))
                loop_time = time()
                loop_count = 0
                # print(encoded_img)

            img_raw = camera.grab()
            if img_raw is not None:
                img_raw = cv2.flip(img_raw, 0)
                resized_img = cv2.resize(img_raw, (640, 480))
                cv2.imshow('Lepton Radiometry [norm+resized]', raw_to_8bit(resized_img))

                loop_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(e)

cv2.destroyAllWindows()
print("Ending...")

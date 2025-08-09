# g1_files/cleanup.py
import sys
import signal
import numpy as np
import pybullet as pb


def install_signal_handler(Config, pybullet_util, comp_time_list=None, video_dir=None, on_exit=None):
    """
    Registra un handler para Ctrl+C que:
      - Guarda tiempos de cómputo (si aplica)
      - Genera video (si aplica)
      - Llama a on_exit() (e.g., plotting)
      - Desconecta de PyBullet y sale
    """
    def _handler(sig, frame):
        try:
            if Config.MEASURE_COMPUTATION_TIME and comp_time_list is not None:
                print("========================================================")
                print('Guardando "computation_time.txt"')
                print("========================================================")
                np.savetxt("computation_time.txt", np.array([comp_time_list]), delimiter=",")
        except Exception as e:
            print(f"[cleanup] Error guardando tiempos: {e}")

        try:
            if Config.VIDEO_RECORD and video_dir is not None:
                print("========================================================")
                print("Generando video")
                print("========================================================")
                pybullet_util.make_video(video_dir)
        except Exception as e:
            print(f"[cleanup] Error generando video: {e}")

        try:
            if callable(on_exit):
                on_exit()
        except Exception as e:
            print(f"[cleanup] Error en on_exit(): {e}")

        pb.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handler)

import os
import sys
import urllib.request
import cv2
import numpy as np


FILES = {
    "prototxt": ("colorization_deploy_v2.prototxt",
                 "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_deploy_v2.prototxt"),
    "caffemodel": ("colorization_release_v2.caffemodel",
                   "https://storage.openvinotoolkit.org/repositories/datumaro/models/colorization/colorization_release_v2.caffemodel"),
    "pts": ("pts_in_hull.npy",
            "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/resources/pts_in_hull.npy"),
}

def ensure_file(local_name, url):
    if os.path.exists(local_name):
        return
    print(f"[Download] {local_name} …")
    urllib.request.urlretrieve(url, local_name)
    if not os.path.exists(local_name) or os.path.getsize(local_name) == 0:
        raise RuntimeError(f"Failed to download {local_name} from {url}")

def pick_image_path():
    
    if len(sys.argv) >= 2:
        return sys.argv[1]
    
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a black & white image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        return path
    except Exception:
        return None

def colorize_image(img_path):
    
    for key, (fname, url) in FILES.items():
        ensure_file(fname, url)

    prototxt = FILES["prototxt"][0]
    caffemodel = FILES["caffemodel"][0]
    pts_path = FILES["pts"][0]

 
    net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
    pts = np.load(pts_path)

    # Add cluster centers to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    img_float = img.astype("float32") / 255.0
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)

    
    resized = cv2.resize(lab, (224, 224))
    L = resized[:, :, 0]
    L -= 50  # mean-centering per model’s training

    
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))  # HxWx2
    ab = cv2.resize(ab, (w, h))

    
    L_orig = lab[:, :, 0]
    color_lab = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    color_bgr = cv2.cvtColor(color_lab, cv2.COLOR_LAB2BGR)
    color_bgr = np.clip(color_bgr, 0, 1)

    
    base, ext = os.path.splitext(img_path)
    out_path = f"{base}_colorized.jpg"
    cv2.imwrite(out_path, (color_bgr * 255).astype("uint8"))
    print(f"[OK] Saved: {out_path}")

    
    try:
        cv2.imshow("Original", img)
        cv2.imshow("Colorized", (color_bgr * 255).astype("uint8"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

if __name__ == "__main__":
    img_path = pick_image_path()
    if not img_path:
        print("Usage: python colorize.py <path_to_image>")
        sys.exit(1)
    colorize_image(img_path)

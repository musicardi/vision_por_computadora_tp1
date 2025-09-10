# ===== LIBRERÍAS =====
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ===== ALGORITMO WHITE PATCH =====
def white_patch_algorithm(img):
    img = img.astype(np.float32)
    max_vals = np.max(img, axis=(0, 1))  # máximo de cada canal
    for c in range(3):
        if max_vals[c] > 0:
            img[:, :, c] = img[:, :, c] / max_vals[c] * 255
    return np.clip(img, 0, 255).astype(np.uint8)

# ===== RUTA DE LA CARPETA =====
carpeta = "white_patch"

# ===== PROCESAR TODAS LAS IMÁGENES =====
for archivo in os.listdir(carpeta):
    if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
        path_in = os.path.join(carpeta, archivo)

        # Leer imagen
        img = cv2.imread(path_in)
        if img is None:
            print(f"⚠️ No se pudo leer: {archivo}")
            continue

        # Aplicar algoritmo
        img_wp = white_patch_algorithm(img)

        # Guardar resultado
        nombre, ext = os.path.splitext(archivo)
        path_out = os.path.join(carpeta, f"{nombre}_wp{ext}")
        cv2.imwrite(path_out, img_wp)

        # Mostrar original y corregida
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_wp_rgb = cv2.cvtColor(img_wp, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.imshow(img_rgb)
        plt.title(f"Original: {archivo}")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(img_wp_rgb)
        plt.title("White Patch")
        plt.axis("off")
        plt.show()

        print(f"✅ Procesada {archivo} -> {nombre}_wp{ext}")

# ===== PARTE 2: HISTOGRAMAS =====
img1_path = "white_patch/img1_tp.png"
img2_path = "white_patch/img2_tp.png"

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("⚠️ No se encontraron img1_tp.png y/o img2_tp.png, revisá las rutas.")
else:
    # Mostrar imágenes
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap="gray")
    plt.title("Imagen 1")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2, cmap="gray")
    plt.title("Imagen 2")
    plt.axis("off")
    plt.show()

    # Calcular histogramas
    bins = 32
    hist1 = cv2.calcHist([img1], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [bins], [0, 256])

    # Graficar histogramas (con ravel() y colores explícitos)
    plt.figure(figsize=(8,4))
    plt.plot(hist1.ravel(), color='blue', label="Hist Img1")
    plt.plot(hist2.ravel(), color='orange', label="Hist Img2")
    plt.title("Comparación de histogramas")
    plt.xlabel("Bins")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()


# ===== PARTE 2: HISTOGRAMAS =====
img1_path = "white_patch/img1_tp.png"
img2_path = "white_patch/img2_tp.png"

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("⚠️ No se encontraron img1_tp.png y/o img2_tp.png, revisá las rutas.")
else:
    # Mostrar imágenes
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(img1, cmap="gray")
    plt.title("Imagen 1")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img2, cmap="gray")
    plt.title("Imagen 2")
    plt.axis("off")
    plt.show()

    # Calcular histogramas
    bins = 256
    hist1 = cv2.calcHist([img1], [0], None, [bins], [0, 256]).flatten()
    hist2 = cv2.calcHist([img2], [0], None, [bins], [0, 256]).flatten()

    x = np.arange(bins)

    plt.figure(figsize=(10,5))
    plt.plot(x, hist1, color='blue', label="Hist Img1")
    plt.plot(x, hist2, color='orange', label="Hist Img2")
    plt.title("Comparación de histogramas")
    plt.xlabel("Nivel de gris (0-255)")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.show()




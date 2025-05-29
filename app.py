import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io
from datetime import datetime


st.set_page_config(
    page_title="ImagePro Visualizer",
    page_icon=":sparkles:",
    layout="centered",
)

st.title("ImagePro :camera:")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded image', width=350)

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    final_img = img_cv.copy()  # Initialize final_img with original image

    # Sidebar options
    option = st.sidebar.selectbox(
        "Choose a processing technique:",
        ["None", "Grayscale", "Gaussian Blur", "Canny Edge Detection", 
         "Dilation", "Erosion", "Opening", "Thresholding", "Laplacian"]
    )

    if option == "Grayscale":
        st.markdown(r"""
        ### 📘 Grayscale Conversion Formula:
        $$ I_{gray} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B $$
        """)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        final_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency
        st.image(gray, caption="Grayscale Image", width=350, channels="GRAY")

    elif option == "Gaussian Blur":
        st.markdown(r"""
        ### 📘 Gaussian Blur Formula:
        $$
        G(x, y) = \frac{1}{2\pi\sigma^2} \cdot e^{ - \frac{x^2 + y^2}{2\sigma^2} }
        $$

        #### 🧩 Example 3×3 Gaussian Kernel:
        $$
        \frac{1}{16}
        \begin{bmatrix}
        1 & 2 & 1 \\
        2 & 4 & 2 \\
        1 & 2 & 1 \\
        \end{bmatrix}
        $$

        - Kernel size (ksize) controls blur strength
        - Must be odd (3, 5, 7, …)
        """)
        ksize = st.sidebar.slider("Adjust Blurness (Odd kernel size)", min_value=1, max_value=31, step=2, value=7)
        blurred = cv2.GaussianBlur(img_cv, (ksize, ksize), 0)
        final_img = blurred
        st.image(blurred, caption=f"Blurred Image (ksize={ksize})", width=350, channels="BGR")

    elif option == "Canny Edge Detection":
        st.markdown(r"""
        ### 📘 Canny Edge Detection Steps:
        1. Apply Gaussian filter to smooth image  
        2. Compute gradient:
        $$ G = \sqrt{G_x^2 + G_y^2} $$
        $$ \theta = \tan^{-1} \left( \frac{G_y}{G_x} \right) $$
        3. Non-maximum suppression  
        4. Hysteresis thresholding using two values
        """)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        threshold1 = st.sidebar.slider("Lower Threshold", 0, 255, 100)
        threshold2 = st.sidebar.slider("Upper Threshold", 0, 255, 200)
        edges = cv2.Canny(gray, threshold1, threshold2)
        final_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency
        st.image(edges, caption="Edge Detected Image", width=350, channels="GRAY")

    elif option == "Dilation":
        st.markdown(rf"""
        ### 📘 Dilation Formula:
        $$ A \oplus B = \{{ z \mid (B)_z \cap A \neq \emptyset \}} $$
        - Structuring element (kernel) size: {kernel_size}×{kernel_size}  
        - Expands white (foreground) regions  
        """)
        kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = cv2.dilate(img_cv, kernel, iterations=1)
        final_img = dilated
        st.image(dilated, caption="Dilated Image", width=350, channels="BGR")

    elif option == "Erosion":
        st.markdown(rf"""
        ### 📘 Erosion Formula:
        $$ A \ominus B = \{{ z \mid (B)_z \subseteq A \}} $$
        - Shrinks white regions  
        - Removes small white noise  
        """)
        kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(img_cv, kernel, iterations=1)
        final_img = eroded
        st.image(eroded, caption="Eroded Image", width=350, channels="BGR")

    elif option == "Opening":
        st.markdown(rf"""
        ### 📘 Opening (Erosion followed by Dilation):
        $$ A \circ B = (A \ominus B) \oplus B $$
        - Removes small white spots (noise)
        """)
        kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        opened = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)
        final_img = opened
        st.image(opened, caption="Opened Image", width=350, channels="BGR")

    elif option == "Thresholding":
        st.markdown(rf"""
        ### 📘 Thresholding Formula:
        $$
        I_{{out}}(x, y) = \begin{{cases}}
        255, & \text{{if  }} I(x, y) > {thresh_val} \\
        0, & \text{{otherwise}}
        \end{{cases}}
        $$
        - Converts grayscale image to binary  
        """)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh_val = st.sidebar.slider("Threshold Value", 0, 255, 128)
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        final_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency
        st.image(thresh, caption="Thresholded Image", width=350, channels="GRAY")

    elif option == "Laplacian":
        st.markdown(r"""
        ### 📘 Laplacian Filter Formula:
        $$
        \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}
        $$

        #### 🔻 Example Kernel:
        $$
        \begin{bmatrix}
        0 & -1 & 0 \\
        -1 & 4 & -1 \\
        0 & -1 & 0 \\
        \end{bmatrix}
        $$
        - Highlights rapid intensity changes in all directions  
        """)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)
        final_img = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for consistency
        st.image(lap, caption="Laplacian Edge Image", width=350, channels="GRAY")

    # Convert final image to RGB for display and download
    final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(final_img_rgb)

    # Compression slider
    quality = st.sidebar.slider("Compression Quality (JPEG)", 10, 100, 90)

    # Compress image to JPEG bytes buffer with chosen quality
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    compressed_img_bytes = buffer.getvalue()

    # Show compressed image preview
    st.image(pil_img, caption=f"Processed Image (Compressed at quality={quality})", width=400)

    # Download button
    date = datetime.now()
    timestamp = date.strftime("%Y-%m-%d_%H-%M-%S")
    st.download_button(
        label="Download Processed Image",
        data=compressed_img_bytes,
        file_name=f"ImagePro{timestamp}.jpg",
        mime="image/jpeg"
    )




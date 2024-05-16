import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Scrollbar, Canvas, Frame
from PIL import Image, ImageTk

def update_image():
    if img is not None:
        display_img = scale_image(img, 800, 600)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)))
        image_label.config(image=img_tk)
        image_label.image = img_tk

def load_image():
    global img, original_img
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        original_img = img.copy()
        update_image()

def apply_filter(filter_func):
    global original_img
    if original_img is not None:
        filtered_img = filter_func(original_img)
        display_image(filtered_img)

def display_image(filtered_img):
    new_window = tk.Toplevel(root)
    new_window.title("Filtered Image")
    new_window.geometry("800x600")
    display_img = scale_image(filtered_img, 800, 600)
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)))
    label = tk.Label(new_window, image=img_tk)
    label.image = img_tk
    label.pack()

    save_button = tk.Button(new_window, text="Download Image", command=lambda: save_image(filtered_img))
    save_button.pack(pady=10)

def save_image(filtered_img):
    file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
    if file_path:
        cv2.imwrite(file_path, filtered_img)

def scale_image(img, max_width, max_height):
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        img = cv2.resize(img, (int(width * scaling_factor), int(height * scaling_factor)))
    return img

def lpf(img):
    return cv2.GaussianBlur(img, (5, 5), 0)

def hpf(img):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -1, kernel)

def mean_filter(img):
    return cv2.blur(img, (5, 5))

def median_filter(img):
    return cv2.medianBlur(img, 5)

def roberts(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 0], [0, -1]], dtype=int)
    kernely = np.array([[0, 1], [-1, 0]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

def prewitt(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
    x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
    y = cv2.filter2D(gray, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

def sobel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def dilation(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def hough_circles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=15, maxRadius=30)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return img

def threshold_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# Placeholder function for region split and merge (complex implementation)
def region_split_and_merge(img):
    return img

# Create the main window
root = tk.Tk()
root.title("Image Filters")
root.geometry("800x600")

# Add a button to load an image
load_button = tk.Button(root, text="Load Image", command=load_image)
load_button.pack(pady=10)

# Add a label to display the image
image_label = tk.Label(root)
image_label.pack()

# Create a frame for the filter buttons
filter_frame = Frame(root)
filter_frame.pack(fill='y', side='right', expand=False)

# Create a canvas to hold the buttons and add a scrollbar
canvas = Canvas(filter_frame)
scrollbar = Scrollbar(filter_frame, orient='vertical', command=canvas.yview)
scrollbar.pack(side='right', fill='y')

canvas.pack(side='left', fill='y', expand=True)
canvas.configure(yscrollcommand=scrollbar.set)

button_frame = Frame(canvas)
canvas.create_window((0, 0), window=button_frame, anchor='nw')

# Add filter buttons
filters = [
    ("Low Pass Filter", lpf),
    ("High Pass Filter", hpf),
    ("Mean Filter", mean_filter),
    ("Median Filter", median_filter),
    ("Roberts Edge Detector", roberts),
    ("Prewitt Edge Detector", prewitt),
    ("Sobel Edge Detector", sobel),
    ("Erosion", erosion),
    ("Dilation", dilation),
    ("Opening", opening),
    ("Closing", closing),
    ("Hough Transform (Circles)", hough_circles),
    ("Threshold Segmentation", threshold_segmentation),
    ("Region Split and Merge", region_split_and_merge)
]

for (filter_name, filter_func) in filters:
    button = tk.Button(button_frame, text=filter_name, command=lambda f=filter_func: apply_filter(f))
    button.pack(pady=5, fill='x')

button_frame.update_idletasks()
canvas.config(scrollregion=canvas.bbox("all"))

# Initialize image variables
img = None
original_img = None

# Start the Tkinter event loop
root.mainloop()

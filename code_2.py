
import tkinter as tk
from tkinter import filedialog, Scrollbar, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Project")
        self.root.geometry("800x600")
        self.root.configure(bg="#E0E0E0")

        self.image_path = None
        self.image = None
        self.filtered_image = None

        self.init_main_window()

    def init_main_window(self):
        self.main_frame = tk.Frame(self.root, bg="#E0E0E0")
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")

        self.title_label = tk.Label(self.main_frame, text="IMAGE PROCESSING PROJECT", font=("Helvetica", 24), bg="#E0E0E0")
        self.title_label.pack(pady=50)

        self.load_image_button = tk.Button(self.main_frame, text="Load Image", font=("Helvetica", 14), command=self.load_image, bg="#404040", fg="#FFFFFF")
        self.load_image_button.pack(pady=20)

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.image = cv2.imread(self.image_path)
            self.show_filter_window()

    def show_filter_window(self):
        self.main_frame.place_forget()

        self.filter_window = tk.Frame(self.root, bg="#E0E0E0")
        self.filter_window.place(relx=0.5, rely=0.5, anchor="center")

        self.filters_label = tk.Label(self.filter_window, text="All Filters", font=("Helvetica", 18), bg="#D0C8B8")
        self.filters_label.grid(row=0, column=0, sticky="ew", padx=10, pady=10)

        self.filters_canvas = Canvas(self.filter_window, bg="#404040")
        self.filters_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        self.scrollbar = Scrollbar(self.filter_window, orient="vertical", command=self.filters_canvas.yview)
        self.scrollbar.grid(row=1, column=1, sticky="ns")

        self.filters_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.filters_frame = tk.Frame(self.filters_canvas, bg="#404040")

        self.filters_canvas.create_window((0, 0), window=self.filters_frame, anchor="nw")

        self.filter_window.grid_rowconfigure(1, weight=1)
        self.filter_window.grid_columnconfigure(0, weight=1)

        self.image_frame = tk.Frame(self.filter_window, bg="#E0E0E0")
        self.image_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=10, pady=10)

        self.back_button = tk.Button(self.image_frame, text="Back", font=("Helvetica", 12), command=self.back_to_main, bg="#404040", fg="#FFFFFF")
        self.back_button.grid(row=0, column=1, sticky="ne", padx=10, pady=10)

        self.filters = [
            "High Pass Filter (HPF)", "Low Pass Filter (LPF)", "Mean Filter", "Median Filter",
            "Roberts Edge Detection", "Prewitt Edge Detection", "Sobel Edge Detection",
            "Erosion", "Dilation", "Open", "Close", "Hough Transform for Circles",
            "Segmentation using Region Split and Merge", "Segmentation using Thresholding"
        ]

        self.filter_buttons = []
        for filter_name in self.filters:
            button = tk.Button(self.filters_frame, text=filter_name, command=lambda f=filter_name: self.apply_filter(f), font=("Helvetica", 12), bg="#B0B0B0", width=40)
            button.pack(pady=5, padx=5)
            self.filter_buttons.append(button)

        self.loaded_image_label = tk.Label(self.image_frame, text="LOADED IMAGE", font=("Helvetica", 18), bg="#E0E0E0")
        self.loaded_image_label.grid(row=1, column=0, sticky="nsew", pady=10)

        self.image_display = tk.Label(self.image_frame)
        self.image_display.grid(row=2, column=0, sticky="nsew", pady=10)

        self.display_image(self.image)

        self.filters_frame.update_idletasks()
        self.filters_canvas.configure(scrollregion=self.filters_canvas.bbox("all"))

    def apply_filter(self, filter_name):
        if filter_name == "High Pass Filter (HPF)":
            self.filtered_image = self.high_pass_filter(self.image)
        elif filter_name == "Low Pass Filter (LPF)":
            self.filtered_image = self.low_pass_filter(self.image)
        elif filter_name == "Mean Filter":
            self.filtered_image = self.mean_filter(self.image)
        elif filter_name == "Median Filter":
            self.filtered_image = self.median_filter(self.image)
        elif filter_name == "Roberts Edge Detection":
            self.filtered_image = self.roberts(self.image)
        elif filter_name == "Prewitt Edge Detection":
            self.filtered_image = self.prewitt(self.image)
        elif filter_name == "Sobel Edge Detection":
            self.filtered_image = self.sobel(self.image)
        elif filter_name == "Erosion":
            self.filtered_image = self.erosion(self.image)
        elif filter_name == "Dilation":
            self.filtered_image = self.dilation(self.image)
        elif filter_name == "Open":
            self.filtered_image = self.opening(self.image)
        elif filter_name == "Close":
            self.filtered_image = self.closing(self.image)
        elif filter_name == "Hough Transform for Circles":
            self.filtered_image = self.hough_circles(self.image)
        elif filter_name == "Segmentation using Region Split and Merge":
            self.filtered_image = self.region_split_and_merge(self.image, 10)
        elif filter_name == "Segmentation using Thresholding":
            self.filtered_image = self.threshold_segmentation(self.image)

        self.show_result_window()

    def high_pass_filter(self, image):
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)

    def low_pass_filter(self, image):
        return cv2.GaussianBlur(image, (5, 5), 0)

    def mean_filter(self, img):
        return cv2.blur(img, (5, 5))

    def median_filter(self, img):
        return cv2.medianBlur(img, 5)

    def roberts(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 0], [0, -1]], dtype=int)
        kernely = np.array([[0, 1], [-1, 0]], dtype=int)
        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

    def prewitt(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=int)
        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

    def sobel(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.cvtColor(cv2.addWeighted(absX, 0.5, absY, 0.5, 0), cv2.COLOR_GRAY2BGR)

    def erosion(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    def dilation(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)

    def opening(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    def closing(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    def hough_circles(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=50, param2=30, minRadius=15, maxRadius=30)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            hough_image = img.copy()
            for (x, y, r) in circles:
                cv2.circle(hough_image, (x, y), r, (0, 255, 0), 4)
                cv2.circle(hough_image, (x, y), 2, (0, 0, 255), 3)
        return hough_image
    
    def region_split_and_merge(self, img, threshold=10):
        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        def split_region(x, y, w, h):
            # If the region is small enough, return it
            if w <= 1 or h <= 1:
                return [(x, y, w, h)]

            region = gray_img[y:y+h, x:x+w]
            mean, stddev = cv2.meanStdDev(region)
            
            if stddev[0] < threshold:
                return [(x, y, w, h)]
            else:
                hw, hh = w // 2, h // 2
                regions = []
                regions.extend(split_region(x, y, hw, hh))
                regions.extend(split_region(x + hw, y, hw, hh))
                regions.extend(split_region(x, y + hh, hw, hh))
                regions.extend(split_region(x + hw, y + hh, hw, hh))
                return regions
        
        def merge_regions(regions):
            merged = np.zeros_like(gray_img)
            for (x, y, w, h) in regions:
                merged[y:y+h, x:x+w] = np.mean(gray_img[y:y+h, x:x+w])
            return merged
        
        # Start with the entire image as the initial region
        initial_regions = split_region(0, 0, gray_img.shape[1], gray_img.shape[0])
        
        # Merge the regions
        merged_image = merge_regions(initial_regions)
        
        return merged_image

    def threshold_segmentation(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


    def show_result_window(self):
        self.filter_window.place_forget()

        self.result_window = tk.Frame(self.root, bg="#E0E0E0")
        self.result_window.place(relx=0.5, rely=0.5, anchor="center")

        self.back_button_result = tk.Button(self.result_window, text="Back", font=("Helvetica", 12), command=self.back_to_filter_window, bg="#404040", fg="#FFFFFF")
        self.back_button_result.grid(row=0, column=1, sticky="ne", padx=10, pady=10)

        self.result_image_label = tk.Label(self.result_window, text="RESULTING IMAGE", font=("Helvetica", 18), bg="#E0E0E0")
        self.result_image_label.grid(row=1, column=0, sticky="nsew", pady=10)

        self.result_image_display = tk.Label(self.result_window)
        self.result_image_display.grid(row=2, column=0, sticky="nsew", pady=10)

        self.display_image(self.filtered_image, label=self.result_image_display)

        self.buttons_frame = tk.Frame(self.result_window, bg="#E0E0E0")
        self.buttons_frame.grid(row=3, column=0, pady=20)

        self.download_button = tk.Button(self.buttons_frame, text="Download Image", command=self.download_image, font=("Helvetica", 14), bg="#404040", fg="#FFFFFF")
        self.download_button.pack(side=tk.LEFT, padx=10)

        self.histogram_button = tk.Button(self.buttons_frame, text="Show Histogram", command=self.show_histogram, font=("Helvetica", 14), bg="#404040", fg="#FFFFFF")
        self.histogram_button.pack(side=tk.RIGHT, padx=10)

    def display_image(self, image, label=None):
        if label is None:
            label = self.image_display

        max_width, max_height = 400, 300  # Set the maximum display size
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        # Scale the image to fit within the maximum display size
        image_pil.thumbnail((max_width, max_height), Image.LANCZOS)
        image_tk = ImageTk.PhotoImage(image_pil)

        label.configure(image=image_tk)
        label.image = image_tk

    def download_image(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            cv2.imwrite(save_path, self.filtered_image)

    def show_histogram(self):
        # Check if the image is already grayscale
        if len(self.filtered_image.shape) == 2:
            # Image is already grayscale
            image_gray = self.filtered_image
        else:
            # Convert to grayscale
            image_gray = cv2.cvtColor(self.filtered_image, cv2.COLOR_BGR2GRAY)
        plt.hist(image_gray.ravel(), 256, [0, 256])
        plt.title("Histogram")
        plt.show()


    def back_to_main(self):
        self.filter_window.place_forget()
        self.init_main_window()

    def back_to_filter_window(self):
        self.result_window.place_forget()
        self.show_filter_window()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()

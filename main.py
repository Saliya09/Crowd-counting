import cv2
import time
import threading
import torch
import numpy as np
import logging
import os
import csv
import socket
import psutil
import ipaddress
from datetime import datetime
from tkinter import messagebox, ttk, filedialog
import tkinter as tk
from PIL import Image, ImageTk
from apscheduler.schedulers.background import BackgroundScheduler
from sort import Sort  # Ensure you have the SORT tracker installed
from fpdf import FPDF  # Install via: pip install fpdf

# =============================================================================
# Global Configuration Variables
# =============================================================================

# --- Logging configuration flags ---
ENABLE_LOGGING = True  # Set to False to disable logging completely
LOG_TO_FILE = True  # If True, log messages will also be written to a file
LOG_LEVEL = logging.DEBUG  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE_NAME = "crowd_count.log"

# --- Timing configuration ---
# Global interval (in minutes) for resetting the cumulative total count.
# Changing INTERVAL_MINUTES will update the reset interval everywhere.
INTERVAL_MINUTES = 30

# --- Window size configuration (1080p aspect ratio) ---
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

# --- CSV file for recording count records for each interval ---
CSV_FILE_NAME = "crowd_count_records.csv"

# --- UI image files (ensure these files exist in the same folder) ---
BACKGROUND_IMAGE_FILE = "background.jpg"
LOGO_IMAGE_FILE = "logo.jpg"

# =============================================================================
# Logging Setup
# =============================================================================
if ENABLE_LOGGING:
    log_handlers = []
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if LOG_TO_FILE:
        try:
            file_handler = logging.FileHandler(LOG_FILE_NAME)
            file_handler.setFormatter(formatter)
            log_handlers.append(file_handler)
        except Exception as e:
            print("Failed to create log file handler:", e)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log_handlers.append(console_handler)

    logging.basicConfig(level=LOG_LEVEL, handlers=log_handlers)
    logger = logging.getLogger(__name__)
else:

    class DummyLogger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def critical(self, *args, **kwargs):
            pass

    logger = DummyLogger()

# =============================================================================
# CSV Setup
# =============================================================================
if not os.path.exists(CSV_FILE_NAME):
    try:
        with open(CSV_FILE_NAME, mode="w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Period", "Count"])
        logger.info("CSV file created: %s", CSV_FILE_NAME)
    except Exception as e:
        logger.error("Failed to create CSV file: %s", e)


def record_csv(start_time, end_time, count):
    """
    Appends a record to the CSV file and automatically generates a PDF for this interval.
    The period is formatted as mmHHddMMyy to mmHHddMMyy.
    """
    try:
        period_str = (
            f"{start_time.strftime('%M%H%d%m%y')} to {end_time.strftime('%M%H%d%m%y')}"
        )
        with open(CSV_FILE_NAME, mode="a", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([period_str, count])
        logger.info("Recorded CSV entry: %s - %d", period_str, count)
        save_interval_pdf(start_time, end_time, count)
    except Exception as e:
        logger.error("Error writing to CSV file: %s", e)


def save_interval_pdf(start_time, end_time, count):
    """
    Converts the current interval's record into a PDF file and saves it automatically.
    The file is named with the period timestamps.
    """
    try:
        period_str = (
            f"{start_time.strftime('%M%H%d%m%y')}_to_{end_time.strftime('%M%H%d%m%y')}"
        )
        pdf_file_name = f"crowd_count_{period_str}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Crowd Counting Record", ln=1, align="C")
        pdf.cell(
            200,
            10,
            txt=f"Period: {start_time.strftime('%M%H%d%m%y')} to {end_time.strftime('%M%H%d%m%y')}",
            ln=2,
            align="C",
        )
        pdf.cell(200, 10, txt=f"Total Count: {count}", ln=3, align="C")
        pdf.output(pdf_file_name)
        logger.info("Automatically saved PDF: %s", pdf_file_name)
    except Exception as e:
        logger.error("Error generating PDF: %s", e)


def download_csv():
    """
    Let the user choose a location to save a copy of the CSV file.
    """
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save CSV File",
        )
        if file_path:
            with open(CSV_FILE_NAME, "r") as src, open(
                file_path, "w", newline=""
            ) as dst:
                dst.write(src.read())
            messagebox.showinfo("CSV Download", "CSV file downloaded successfully!")
    except Exception as e:
        logger.error("Error downloading CSV file: %s", e)
        messagebox.showerror("Error", "Failed to download CSV file.")


def download_pdf():
    """
    Converts the entire CSV file into a PDF file and lets the user save it.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Crowd Counting Records", ln=1, align="C")
        pdf.ln(10)
        with open(CSV_FILE_NAME, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                line = " | ".join(row)
                pdf.cell(200, 10, txt=line, ln=1, align="L")
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save PDF File",
        )
        if file_path:
            pdf.output(file_path)
            messagebox.showinfo("PDF Download", "PDF file downloaded successfully!")
    except Exception as e:
        logger.error("Error downloading PDF file: %s", e)
        messagebox.showerror("Error", "Failed to download PDF file.")


# =============================================================================
# Model and Tracker Initialization
# =============================================================================
try:
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    logger.info("YOLOv5 model loaded successfully.")
except Exception as e:
    logger.error("Error loading YOLOv5 model: %s", e)
    raise

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# =============================================================================
# Global Variables for Counting and Stream Control
# =============================================================================
global_total_count = 0  # Cumulative count for the current interval period
current_frame_count = 0  # Count of persons in the current frame
previous_ids = set()  # Set of object IDs from the previous frame
last_reset_time = time.time()  # Timestamp when the current interval began
period_start_time = datetime.fromtimestamp(last_reset_time)

ENABLE_ID_DISPLAY = True

stream_threads = {}
stop_stream_flags = {}  # key: RTSP URL, value: threading.Event()


# =============================================================================
# Network Scanning Functions
# =============================================================================
def get_all_networks():
    networks = []
    for interface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                ip = addr.address
                try:
                    network = ipaddress.IPv4Network(ip + "/24", strict=False)
                    networks.append(network)
                except ipaddress.AddressValueError:
                    pass
    return networks


def scan_port(ip, port=554, timeout=0.5):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((ip, port))
        s.close()
        return ip
    except Exception:
        s.close()
        return None


def scan_ip_addresses():
    networks = get_all_networks()
    available_ips = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for network in networks:
            for ip in network.hosts():
                futures.append(executor.submit(scan_port, str(ip)))
        for future in as_completed(futures):
            result = future.result()
            if result:
                available_ips.append(result)
    return available_ips


# =============================================================================
# VideoCapture Class
# =============================================================================
class VideoCapture:
    def __init__(self, source):
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                logger.error("Failed to open video source: %s", source)
                raise Exception("Video source open failed.")
            self.latest_frame = None
            self.lock = threading.Lock()
            self._stopped = False
            t = threading.Thread(target=self._reader, daemon=True)
            t.start()
            logger.info("VideoCapture initialized for source: %s", source)
        except Exception as e:
            logger.error("Error initializing VideoCapture: %s", e)
            raise

    def _reader(self):
        while not self._stopped:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Frame read failed; retrying...")
                    time.sleep(0.1)
                    continue
                with self.lock:
                    self.latest_frame = frame
            except Exception as e:
                logger.error("Error reading frame: %s", e)
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.latest_frame is not None:
                try:
                    return self.latest_frame.copy()
                except Exception as e:
                    logger.error("Error copying frame: %s", e)
                    return None
            return None

    def stop(self):
        try:
            self._stopped = True
            self.cap.release()
            logger.info("VideoCapture stopped.")
        except Exception as e:
            logger.error("Error stopping VideoCapture: %s", e)


# =============================================================================
# Crowd Counting Frame Processing
# =============================================================================
def process_frame(frame):
    global global_total_count, previous_ids, last_reset_time, period_start_time, current_frame_count, INTERVAL_MINUTES
    try:
        results = model(frame)
    except Exception as e:
        logger.error("Model inference error: %s", e)
        return frame

    try:
        detections = results.xyxy[0].cpu().numpy()
    except Exception as e:
        logger.error("Error processing model results: %s", e)
        return frame

    person_detections = [d for d in detections if int(d[5]) == 0]
    if len(person_detections) > 0:
        detections_for_sort = np.array(
            [[x1, y1, x2, y2, conf] for x1, y1, x2, y2, conf, cls in person_detections]
        )
    else:
        detections_for_sort = np.empty((0, 5))

    try:
        tracked_objects = tracker.update(detections_for_sort)
    except Exception as e:
        logger.error("Tracker update error: %s", e)
        tracked_objects = []

    current_ids = set()
    for obj in tracked_objects:
        try:
            x1, y1, x2, y2, obj_id = [int(coord) for coord in obj]
            current_ids.add(obj_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if ENABLE_ID_DISPLAY:
                cv2.putText(
                    frame,
                    f"ID: {obj_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
        except Exception as e:
            logger.error("Error processing tracked object: %s", e)

    current_frame_count = len(current_ids)
    new_appearances = current_ids - previous_ids
    global_total_count += len(new_appearances)
    previous_ids = current_ids

    # Check if the interval has elapsed.
    current_time = time.time()
    if current_time - last_reset_time >= INTERVAL_MINUTES * 60:
        period_end_time = datetime.now()
        record_csv(period_start_time, period_end_time, global_total_count)
        global_total_count = 0
        previous_ids = set()
        last_reset_time = current_time
        period_start_time = datetime.fromtimestamp(last_reset_time)

    cv2.putText(
        frame,
        f"Current Count: {current_frame_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Total Count ({INTERVAL_MINUTES} min): {global_total_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    return frame


# =============================================================================
# Streaming Function (Tkinter UI for RTSP stream)
# =============================================================================
def stream_rtsp(rtsp_url, stream_window):
    stop_flag = threading.Event()
    stop_stream_flags[rtsp_url] = stop_flag

    # Set up background image for streaming window.
    try:
        bg_img = Image.open(BACKGROUND_IMAGE_FILE)
    except Exception as e:
        logger.error("Error loading background image: %s", e)
        bg_img = None

    stream_window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    stream_window.title(f"Stream: {rtsp_url}")
    if bg_img:
        bg_resized = bg_img.resize((WINDOW_WIDTH, WINDOW_HEIGHT))
        bg_photo = ImageTk.PhotoImage(bg_resized)
        bg_label = tk.Label(stream_window, image=bg_photo)
        bg_label.image = bg_photo
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    header_frame = ttk.Frame(stream_window)
    header_frame.pack(fill=tk.X, pady=5)
    header_label = ttk.Label(
        header_frame,
        text="Rezler Systems Stream Viewer",
        font=("Helvetica", 18, "bold"),
    )
    header_label.pack(side=tk.LEFT, padx=10)
    footer_frame = ttk.Frame(stream_window)
    footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    footer_label = ttk.Label(
        footer_frame, text="© Rezler Systems Private Ltd", font=("Helvetica", 10)
    )
    footer_label.pack(side=tk.RIGHT, padx=10)

    content_frame = ttk.Frame(stream_window)
    content_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
    video_label = ttk.Label(content_frame)
    video_label.pack(pady=10)
    current_count_label = ttk.Label(
        content_frame, text="Current Count: 0", font=("Helvetica", 16)
    )
    current_count_label.pack(pady=5)
    total_count_label = ttk.Label(
        content_frame,
        text=f"Total Count ({INTERVAL_MINUTES} min): 0",
        font=("Helvetica", 16),
    )
    total_count_label.pack(pady=5)
    # Download button now offers both CSV and PDF options.
    download_frame = ttk.Frame(content_frame)
    download_frame.pack(pady=5)
    ttk.Button(download_frame, text="Download CSV", command=download_csv).pack(
        side=tk.LEFT, padx=5
    )
    ttk.Button(download_frame, text="Download PDF", command=download_pdf).pack(
        side=tk.LEFT, padx=5
    )

    try:
        cap = VideoCapture(rtsp_url)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open stream at {rtsp_url}")
        stream_window.destroy()
        return

    frame_delay = 1 / 30

    try:
        while not stop_flag.is_set():
            frame = cap.read()
            if frame is None:
                time.sleep(0.1)
                continue

            processed = process_frame(frame)
            try:
                frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                img_pil = img_pil.resize((800, 600))
                imgtk = ImageTk.PhotoImage(image=img_pil)
            except Exception as e:
                logger.error("Error converting frame for display: %s", e)
                continue

            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            current_count_label.configure(text=f"Current Count: {current_frame_count}")
            total_count_label.configure(
                text=f"Total Count ({INTERVAL_MINUTES} min): {global_total_count}"
            )

            stream_window.update_idletasks()
            time.sleep(frame_delay)
    except Exception as e:
        logger.error("Exception in stream thread for %s: %s", rtsp_url, e)
    finally:
        cap.stop()
        stop_stream_flags[rtsp_url].set()
        stream_window.destroy()
        logger.info("Stream for %s stopped.", rtsp_url)


def start_stream(rtsp_url):
    stream_window = tk.Toplevel()
    thread = threading.Thread(target=stream_rtsp, args=(rtsp_url, stream_window))
    thread.daemon = True
    thread.start()
    stream_threads[rtsp_url] = thread


# =============================================================================
# Tkinter UI Functions for Scanning and Credential Entry
# =============================================================================
def show_credentials_frame(selected_ips, root):
    credentials_window = tk.Toplevel(root)
    credentials_window.title("Enter Camera Connection Details")
    credentials_window.geometry("500x600")

    try:
        bg_img = Image.open(BACKGROUND_IMAGE_FILE)
        bg_resized = bg_img.resize((500, 600))
        bg_photo = ImageTk.PhotoImage(bg_resized)
        bg_label = tk.Label(credentials_window, image=bg_photo)
        bg_label.image = bg_photo
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        logger.error("Error loading background for credentials window: %s", e)

    credentials_dict = {}
    for ip in selected_ips:
        frame = ttk.Frame(credentials_window, padding="10")
        frame.pack(pady=5, fill="x")
        ttk.Label(frame, text=f"Camera at {ip}", font=("Helvetica", 12, "bold")).pack(
            anchor="w"
        )
        ttk.Label(frame, text="Username:").pack(anchor="w")
        username_entry = ttk.Entry(frame, width=30)
        username_entry.pack(anchor="w")
        ttk.Label(frame, text="Password:").pack(anchor="w")
        password_entry = ttk.Entry(frame, width=30, show="*")
        password_entry.pack(anchor="w")
        # Dynamic RTSP URL entry
        ttk.Label(frame, text="RTSP URL:").pack(anchor="w")
        # Suggest a default RTSP URL template, but allow editing
        default_rtsp = f"rtsp://{{username}}:{{password}}@{ip}:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"
        rtsp_entry = ttk.Entry(frame, width=60)
        rtsp_entry.insert(0, default_rtsp)
        rtsp_entry.pack(anchor="w")
        credentials_dict[ip] = (username_entry, password_entry, rtsp_entry)

    def start_streams():
        for ip, (
            username_entry,
            password_entry,
            rtsp_entry,
        ) in credentials_dict.items():
            username = username_entry.get().strip()
            password = password_entry.get().strip()
            rtsp_url_template = rtsp_entry.get().strip()
            if not username or not password or not rtsp_url_template:
                messagebox.showerror(
                    "Missing Information", f"Please enter all fields for {ip}"
                )
                return
            # Replace placeholders if present
            rtsp_url = (
                rtsp_url_template.replace("{username}", username)
                .replace("{password}", password)
                .replace("{ip}", ip)
            )
            start_stream(rtsp_url)
        credentials_window.destroy()

    ttk.Button(credentials_window, text="Start Streams", command=start_streams).pack(
        pady=20
    )


def handle_next(ip_listbox, root):
    selected_indices = ip_listbox.curselection()
    if not selected_indices:
        messagebox.showinfo("No Selection", "Please select at least one IP address.")
        return
    selected_ips = [ip_listbox.get(i) for i in selected_indices]
    show_credentials_frame(selected_ips, root)


def scan_ips(ip_listbox, scanning_label, next_button):
    def run_scan():
        scanning_label.config(text="Scanning networks for devices...")
        ip_listbox.delete(0, tk.END)
        try:
            available_ips = scan_ip_addresses()
        except Exception as e:
            logger.error("Error scanning IP addresses: %s", e)
            available_ips = []

        def update_listbox():
            for ip in available_ips:
                ip_listbox.insert(tk.END, ip)
            scanning_label.config(text="Scan complete.")
            next_button.config(state=tk.NORMAL)

        ip_listbox.after(0, update_listbox)

    threading.Thread(target=run_scan, daemon=True).start()


# =============================================================================
# Main Tkinter Application UI (Entry Page)
# =============================================================================
def main_app():
    global INTERVAL_MINUTES
    root = tk.Tk()
    root.title("Crowd Counting Stream Viewer")
    root.geometry("1400x800")

    try:
        bg_img = Image.open(BACKGROUND_IMAGE_FILE)
        bg_img = bg_img.resize((1400, 800))
        bg_photo = ImageTk.PhotoImage(bg_img)
        bg_label = tk.Label(root, image=bg_photo)
        bg_label.image = bg_photo
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    except Exception as e:
        logger.error("Error loading main window background: %s", e)

    header_frame = ttk.Frame(root, padding="10")
    header_frame.pack(fill=tk.X, pady=10)
    try:
        logo_img = Image.open(LOGO_IMAGE_FILE)
        logo_img = logo_img.resize((100, 100), Image.LANCZOS)
        logo_photo = ImageTk.PhotoImage(logo_img)
        logo_label = ttk.Label(header_frame, image=logo_photo)
        logo_label.image = logo_photo
        logo_label.pack(side=tk.LEFT, padx=10)
    except Exception as e:
        logger.error("Error loading logo image: %s", e)
    header_title = ttk.Label(
        header_frame,
        text="Rezler Systems Stream Viewer",
        font=("Helvetica", 24, "bold"),
    )
    header_title.pack(side=tk.LEFT, padx=10)

    interval_frame = ttk.Frame(root, padding="10")
    interval_frame.pack(pady=10)
    ttk.Label(
        interval_frame, text="Reset Interval (minutes):", font=("Helvetica", 14)
    ).pack(side=tk.LEFT, padx=5)
    interval_entry = ttk.Entry(interval_frame, width=5, font=("Helvetica", 14))
    interval_entry.insert(0, str(INTERVAL_MINUTES))
    interval_entry.pack(side=tk.LEFT, padx=5)

    def update_interval():
        global INTERVAL_MINUTES
        try:
            new_val = int(interval_entry.get())
            INTERVAL_MINUTES = new_val
            messagebox.showinfo(
                "Interval Updated", f"Interval set to {INTERVAL_MINUTES} minutes."
            )
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer.")

    ttk.Button(interval_frame, text="Update Interval", command=update_interval).pack(
        side=tk.LEFT, padx=5
    )

    ip_frame = ttk.Frame(root, padding="10")
    ip_frame.pack(pady=10)
    ip_listbox = tk.Listbox(
        ip_frame, selectmode=tk.MULTIPLE, width=50, font=("Helvetica", 12)
    )
    ip_listbox.pack(side=tk.LEFT, padx=5)
    scrollbar = ttk.Scrollbar(ip_frame, orient=tk.VERTICAL, command=ip_listbox.yview)
    scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    ip_listbox.config(yscrollcommand=scrollbar.set)

    next_button = ttk.Button(
        root, text="Next", command=lambda: handle_next(ip_listbox, root)
    )
    next_button.pack(pady=20)
    next_button.config(state=tk.DISABLED)

    scanning_label = ttk.Label(root, text="Scanning...", font=("Helvetica", 12))
    scanning_label.pack(pady=5)
    ttk.Button(
        root,
        text="Scan IP Addresses",
        command=lambda: scan_ips(ip_listbox, scanning_label, next_button),
    ).pack(pady=10)

    footer_frame = ttk.Frame(root, padding="10")
    footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
    footer_label = ttk.Label(
        footer_frame, text="© Rezler Systems Private Ltd", font=("Helvetica", 10)
    )
    footer_label.pack(side=tk.RIGHT, padx=10)

    root.after(100, lambda: scan_ips(ip_listbox, scanning_label, next_button))
    root.mainloop()


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    main_app()

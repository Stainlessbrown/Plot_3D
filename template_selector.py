import tkinter as tk
from tkinter import filedialog
import os
import logging

class TemplateSelector:
    def __init__(self):
        self.file_path = None
        self.root = None
        self.create_and_run_dialog()
        
    def create_and_run_dialog(self):
        """Create and run the file selection dialog with proper error handling."""
        try:
            # Create the main window
            self.root = tk.Tk()
            self.root.title("Select Worksheet")
            self.root.geometry("300x100")
            self.root.lift()
            self.root.attributes("-topmost", True)
            
            # Add heading
            heading = tk.Label(self.root, text=".ods", font=("Arial", 14, "bold"))
            heading.pack(pady=10)
            
            custom_button = tk.Button(
                self.root, 
                text="Select File",
                command=self.select_custom_file,
                width=20,
                height=2
            )
            custom_button.pack(pady=5)
            
            # Start the main loop
            self.root.mainloop()
        except Exception as e:
            logging.error(f"Error creating template selector window: {str(e)}")
            if self.root:
                try:
                    self.root.destroy()
                except:
                    pass
            raise
    
    def select_custom_file(self):
        logging.debug("Select File")
        # Create a new temporary tkinter root for the file dialog
        file_root = tk.Tk()
        file_root.withdraw()
        
        file_types = [
            ('OpenDocument Spreadsheet', '*.ods'),
            ('All files', '*.*')
        ]
        
        selected_file = filedialog.askopenfilename(filetypes=file_types)
        file_root.destroy()
        
        if selected_file:
            # Convert to absolute path
            self.file_path = os.path.abspath(selected_file)
            logging.info(f"Selected file (absolute path): {self.file_path}")
            self.root.destroy()
        else:
            logging.warning("No file selected")


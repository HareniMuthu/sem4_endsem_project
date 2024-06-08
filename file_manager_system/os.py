import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, scrolledtext, ttk
import os
import shutil
import sqlite3
from cryptography.fernet import Fernet
from datetime import datetime


# Defining a global password
PASSWORD = "Surya1"
SECRET_DIRECTORY = ".my_secret_dir"  # Hidden directory

class FileManagerApp:
    def _init_(self, root, username):
        self.root = root
        root.title("Python File Manager")
        self.current_path = os.getcwd()
        self.history = [self.current_path]  # Initialize directory history
        self.username = username
        self.init_gui_components()
        self.create_secret_directory()
        self.setup_database()
        self.show_welcome_message()
        self.theme = "default"  # Default theme

    def init_gui_components(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(self.frame)
        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=self.tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.configure(yscrollcommand=self.scrollbar.set)

        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD)
        self.text_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.populate_tree()

    def create_secret_directory(self):
        secret_dir = os.path.join(self.current_path, SECRET_DIRECTORY)
        if not os.path.exists(secret_dir):
            os.makedirs(secret_dir, exist_ok=True)

    def setup_database(self):
        self.conn = sqlite3.connect(os.path.join(SECRET_DIRECTORY, "user_credentials.db"))
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
        self.conn.commit()

    def populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for entry in os.listdir(self.current_path):
            path = os.path.join(self.current_path, entry)
            if os.path.isdir(path):
                self.tree.insert('', 'end', text=entry, values=(path, 'Folder'))
            else:
                self.tree.insert('', 'end', text=entry, values=(path, 'File'))


    def get_creation_time(self, file_path):
        try:
            creation_timestamp = os.path.getctime(file_path)
            creation_time = datetime.fromtimestamp(creation_timestamp)
            return creation_time
        except FileNotFoundError:
            return None

    def get_file_metadata(self, file_path):
        try:
            size = os.path.getsize(file_path)  # File size in bytes
            created_time = os.path.getctime(file_path)  # Creation time
            modified_time = os.path.getmtime(file_path)  # Modification time

            # Convert creation and modification times to datetime objects
            created_time_dt = datetime.fromtimestamp(created_time)
            modified_time_dt = datetime.fromtimestamp(modified_time)

            # Format dates
            formatted_creation_time = created_time_dt.strftime("%Y-%m-%d %H:%M:%S")
            formatted_modified_time = modified_time_dt.strftime("%Y-%m-%d %H:%M:%S")

            metadata = f"File Size: {size} bytes\n"
            metadata += f"Created Time: {formatted_creation_time}\n"
            metadata += f"Modified Time: {formatted_modified_time}"

            return metadata
        except FileNotFoundError:
            return f"Error: File not found: {file_path}"

    def display_metadata(self):
        selected_items = self.tree.selection()
        if selected_items:
            selected_item = selected_items[0]
            file_path = self.tree.item(selected_item, 'values')[0]
            metadata = self.get_file_metadata(file_path)
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.END, metadata)
        else:
            messagebox.showwarning("Warning", "Please select a file to display metadata.")

    def create_file(self):
        # Ask for password before creating a file
        password = simpledialog.askstring("Password", "Enter password:")
        if password == PASSWORD:
            file_name = filedialog.asksaveasfilename(initialdir=self.current_path)
            if file_name:
                open(file_name, 'a').close()
                self.populate_tree()
        else:
            messagebox.showerror("Authentication Failed", "Incorrect password!")

    def create_folder(self):
        # Ask for password before creating a folder
        password = simpledialog.askstring("Password", "Enter password:")
        if password == PASSWORD:
            folder_name = simpledialog.askstring("Folder Name", "Enter the name for the new folder:", parent=self.root)
            if folder_name:
                os.makedirs(os.path.join(self.current_path, folder_name), exist_ok=True)
                self.populate_tree()
        else:
            messagebox.showerror("Authentication Failed", "Incorrect password!")

    def delete_file_or_folder(self):
        # Asking for password before deleting a file or folder
        password = simpledialog.askstring("Password", "Enter password:")
        if password == PASSWORD:
            selected_items = self.tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                file_path = self.tree.item(selected_item, 'values')[0]
                if messagebox.askyesno("Delete", f"Do you want to delete '{file_path}'?"):
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # To delete non-empty directories
                    else:
                        os.remove(file_path)
                    self.populate_tree()
            else:
                messagebox.showwarning("Warning", "Please select a file or folder to delete.")
        else:
            messagebox.showerror("Authentication Failed", "Incorrect password!")

    def change_directory(self, path=None):
        if path:
            self.current_path = path
        else:
            path = filedialog.askdirectory(initialdir=self.current_path)
            if path:
                self.current_path = path
                self.history.append(self.current_path)  # Update history
        self.populate_tree()

    def open_file(self):
        # Asking for password before opening a file
        password = simpledialog.askstring("Password", "Enter password:")
        if password == PASSWORD:
            selected_items = self.tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                file_path = self.tree.item(selected_item, 'values')[0]
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r') as file:
                            self.text_area.delete(1.0, tk.END)
                            self.text_area.insert(tk.END, file.read())
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                elif os.path.isdir(file_path):
                    self.change_directory(file_path)
            else:
                messagebox.showwarning("Warning", "Please select a file or folder to open.")
        else:
            messagebox.showerror("Authentication Failed", "Incorrect password!")

    def save_file(self):
        # Asking for password before saving a file
        password = simpledialog.askstring("Password", "Enter password:")
        if password == PASSWORD:
            selected_items = self.tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                file_path = self.tree.item(selected_item, 'values')[0]
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'w') as file:
                            file.write(self.text_area.get(1.0, tk.END))
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
        else:
            messagebox.showerror("Authentication Failed", "Incorrect password!")

    def go_back(self):
        if len(self.history) > 1:
            self.history.pop()  # Remove current directory
            previous_path = self.history[-1]  # Get the previous directory
            self.change_directory(previous_path)

    def show_welcome_message(self):
        welcome_label = tk.Label(self.root, text=f"Welcome, {self.username}!", font=("Helvetica", 12))
        welcome_label.pack(side=tk.TOP, padx=10, pady=10)

    def change_theme(self):
        theme_choice = simpledialog.askstring("Theme Change", "Choose theme (light/dark):")
        if theme_choice and theme_choice.lower() == "dark":
            self.theme = "dark"
            self.set_dark_theme()
        elif theme_choice and theme_choice.lower() == "light":
            self.theme = "light"
            self.set_light_theme()
        else:
            messagebox.showerror("Invalid Theme", "Please choose a valid theme (light/dark).")

    def set_dark_theme(self):
        self.root.config(bg="#22223b")
        self.frame.config(bg="#22223b")
        self.tree.config(style="Dark.Treeview")
        self.text_area.config(bg="#22223b", fg="white")
        self.set_button_theme("red")

    def set_light_theme(self):
        self.root.config(bg="#8ecae6")
        self.frame.config(bg="#219ebc")
        self.tree.config(style="Light.Treeview")
        self.text_area.config(bg="#8ecae6", fg="black")
        self.set_button_theme("red")

    def set_button_theme(self, button_bg):
        style = ttk.Style()
        style.configure("TButton", background=button_bg)

    def generate_encryption_key(self):
        key_file_path = os.path.join(SECRET_DIRECTORY, "encryption_key.key")
        if not os.path.exists(key_file_path):
            key = Fernet.generate_key()
            with open(key_file_path, 'wb') as key_file:
                key_file.write(key)

    def encrypt_selected_file(self):
        password = simpledialog.askstring("Password", "Enter password:")
        if password == PASSWORD:
            selected_items = self.tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                file_path = self.tree.item(selected_item, 'values')[0]
                if os.path.isfile(file_path):
                    key_file_path = os.path.join(SECRET_DIRECTORY, "encryption_key.key")
                    output_file_path = file_path + ".enc"
                    self.generate_encryption_key()
                    try:
                        with open(key_file_path, 'rb') as key_file:
                            key = key_file.read()

                        cipher_suite = Fernet(key)

                        with open(file_path, 'rb') as file:
                            file_data = file.read()

                        encrypted_data = cipher_suite.encrypt(file_data)

                        with open(output_file_path, 'wb') as encrypted_file:
                            encrypted_file.write(encrypted_data)
                        self.populate_tree()
                        messagebox.showinfo("Encryption Successful", f"File '{file_path}' encrypted.")
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                else:
                    messagebox.showwarning("Warning", "Please select a file to encrypt.")
            else:
                messagebox.showwarning("Warning", "Please select a file to encrypt.")
        else:
            messagebox.showerror("Authentication Failed", "Incorrect password!")

    def decrypt_selected_file(self):
        password = simpledialog.askstring("Password", "Enter password:")
        if password == PASSWORD:
            selected_items = self.tree.selection()
            if selected_items:
                selected_item = selected_items[0]
                file_path = self.tree.item(selected_item, 'values')[0]
                if os.path.isfile(file_path) and file_path.endswith(".enc"):
                    key_file_path = os.path.join(SECRET_DIRECTORY, "encryption_key.key")
                    output_file_path = file_path[:-4] + ".decrypted"  # Remove the .enc extension
                    try:
                        with open(key_file_path, 'rb') as key_file:
                            key = key_file.read()

                        cipher_suite = Fernet(key)

                        with open(file_path, 'rb') as encrypted_file:
                            encrypted_data = encrypted_file.read()

                        decrypted_data = cipher_suite.decrypt(encrypted_data)

                        with open(output_file_path, 'wb') as decrypted_file:
                            decrypted_file.write(decrypted_data)
                        self.populate_tree()
                        messagebox.showinfo("Decryption Successful", f"File '{file_path}' decrypted.")
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                else:
                    messagebox.showwarning("Warning", "Please select an encrypted file to decrypt.")
            else:
                messagebox.showwarning("Warning", "Please select an encrypted file to decrypt.")
        else:
            messagebox.showerror("Authentication Failed", "Incorrect password!")

    def setup_layout(self):
        button_frame = ttk.LabelFrame(self.root, text="Actions")
        button_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        button_row1_frame = tk.Frame(button_frame)
        button_row1_frame.pack(side=tk.TOP, fill=tk.X)

        create_file_btn = tk.Button(button_row1_frame, text="Create File", command=self.create_file)
        create_file_btn.pack(side=tk.LEFT, padx=5, pady=5)

        display_metadata_btn = tk.Button(button_row1_frame, text="Display Metadata", command=self.display_metadata)
        display_metadata_btn.pack(side=tk.LEFT, padx=5, pady=5)

        create_folder_btn = tk.Button(button_row1_frame, text="Create Folder", command=self.create_folder)
        create_folder_btn.pack(side=tk.LEFT, padx=5, pady=5)

        open_file_btn = tk.Button(button_row1_frame, text="Open File", command=self.open_file)
        open_file_btn.pack(side=tk.LEFT, padx=5, pady=5)

        save_btn = tk.Button(button_row1_frame, text="Save File", command=self.save_file)
        save_btn.pack(side=tk.LEFT, padx=5, pady=5)

        button_row2_frame = tk.Frame(button_frame)
        button_row2_frame.pack(side=tk.TOP, fill=tk.X)

        delete_btn = tk.Button(button_row2_frame, text="Delete", command=self.delete_file_or_folder)
        delete_btn.pack(side=tk.LEFT, padx=5, pady=5)

        back_btn = tk.Button(button_row2_frame, text="Back", command=self.go_back)
        back_btn.pack(side=tk.LEFT, padx=5, pady=5)

        change_dir_btn = tk.Button(button_row2_frame, text="Change Directory", command=self.change_directory)
        change_dir_btn.pack(side=tk.LEFT, padx=5, pady=5)

        theme_btn = tk.Button(button_row2_frame, text="Change Theme", command=self.change_theme)
        theme_btn.pack(side=tk.LEFT, padx=5)


        button_row3_frame = tk.Frame(button_frame)
        button_row3_frame.pack(side=tk.TOP, fill=tk.X)

        encrypt_btn = tk.Button(button_row3_frame, text="Encrypt File", command=self.encrypt_selected_file)
        encrypt_btn.pack(side=tk.LEFT, padx=5, pady=5)

        decrypt_btn = tk.Button(button_row3_frame, text="Decrypt File", command=self.decrypt_selected_file)
        decrypt_btn.pack(side=tk.LEFT, padx=5, pady=5)

    def run(self):
        self.setup_layout()
        self.root.mainloop()

    def delete(self):
        self.conn.close()

def main():
    root = tk.Tk()
    root.withdraw()  

    # Ask for username and password
    username = simpledialog.askstring("Login", "Enter your username:")
    password = simpledialog.askstring("Login", "Enter your password:", show='*')

    # Check if the username and password are correct
    if username and password == PASSWORD:
        root.deiconify()  # Show the root window
        app = FileManagerApp(root, username)
        app.run()
    else:
        messagebox.showerror("Authentication Failed", "Incorrect username or password!")
        root.destroy()  # To Close the application

if _name_ == "_main_":
    main()
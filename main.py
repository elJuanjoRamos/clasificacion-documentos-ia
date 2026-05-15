from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import customtkinter as ctk
from PIL import Image, ImageTk


"""CLASIFICADOR"""
from backend.procesador import process_documents


ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".doc": "Word",
    ".xlsx": "Excel",
    ".xls": "Excel",
    ".png": "Imagen",
    ".jpg": "Imagen",
    ".jpeg": "Imagen",
    ".tif": "Imagen",
    ".tiff": "Imagen",
    ".bmp": "Imagen",
    ".xml": "XML",
    ".txt": "TXT"
}


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def scan_folder(folder_path: Path, progress_callback=None) -> list[dict[str, str]]:
    all_files = [p for p in folder_path.rglob("*") if p.is_file()]
    total_candidates = len(all_files) if all_files else 1

    results: list[dict[str, str]] = []

    for index, path in enumerate(all_files, start=1):
        ext = path.suffix.lower()

        if ext in SUPPORTED_EXTENSIONS:
            try:
                stat = path.stat()
                results.append(
                    {
                        "name": path.name,
                        "type": SUPPORTED_EXTENSIONS[ext],
                        "extension": ext,
                        "size": format_size(stat.st_size),
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                        "full_path": str(path),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "name": path.name,
                        "type": "Error",
                        "extension": ext,
                        "size": "-",
                        "modified": "-",
                        "full_path": f"{path} | Error: {e}",
                    }
                )

        if progress_callback:
            progress_callback(index / total_candidates)

    return results


class DocumentIngestionApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Demo de ingesta documental")
        self.geometry("1500x860")
        self.minsize(1250, 760)

        self.selected_folder = ""
        self.tree_images: dict[str, ImageTk.PhotoImage] = {}

        self._load_icons()
        self._build_ui()

    def _load_icons(self) -> None:
        def load_icon(path: str, size=(30, 30)):
            return ctk.CTkImage(Image.open(path), size=size)

        self.ctk_icons = {
            "folder":   load_icon("icons/folder.png"),
            "play":     load_icon("icons/play.png"),
            "file":     load_icon("icons/file.png"),
            "pdf":      load_icon("icons/pdf.png"),
            "image":    load_icon("icons/image.png"),
            "excel":    load_icon("icons/excel.png"),
            "clock":    load_icon("icons/clock.png"),
            "word":     load_icon("icons/word.png"),
            "xml":      load_icon("icons/xml.png"),
            "txt":      load_icon("icons/txt.png"),
            "categoria":load_icon("icons/categoria.png")
        }

        def load_tree_icon(path: str, size=(18, 18)):
            img = Image.open(path).resize(size, Image.LANCZOS)
            return ImageTk.PhotoImage(img)

        self.tree_images = {
            "PDF":      load_tree_icon("icons/pdf.png"),
            "TXT":      load_tree_icon("icons/txt.png"),
            "XML":      load_tree_icon("icons/xml.png"),
            "Word":     load_tree_icon("icons/word.png"),
            "Excel":    load_tree_icon("icons/excel.png"),
            "Imagen":   load_tree_icon("icons/image.png"),
            "Error":    load_tree_icon("icons/file.png"),
            "default":  load_tree_icon("icons/file.png"),
        }

    def _build_ui(self) -> None:
        self.configure(fg_color="#f4f4f4")

        ctk.CTkLabel(
            self,
            text="Carpeta seleccionada:",
            font=("Segoe UI", 15, "bold"),
            text_color="#1f1f1f",
        ).pack(anchor="w", padx=20, pady=(18, 8))

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="x", padx=20)

        self.folder_entry = ctk.CTkEntry(
            top_frame,
            height=42,
            font=("Segoe UI", 14),
            corner_radius=8,
        )
        self.folder_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.select_button = ctk.CTkButton(
            top_frame,
            text="Seleccionar carpeta",
            image=self.ctk_icons["folder"],
            compound="left",
            height=42,
            corner_radius=8,
            font=("Segoe UI", 14),
            command=self.select_folder,
            fg_color="#efefef",
            text_color="#1f1f1f",
            hover_color="#e4e4e4",
            border_width=1,
            border_color="#d0d0d0",
        )
        self.select_button.pack(side="left", padx=(0, 16))

        self.scan_button = ctk.CTkButton(
            top_frame,
            text="Escanear",
            image=self.ctk_icons["play"],
            compound="left",
            height=42,
            corner_radius=8,
            font=("Segoe UI", 14, "bold"),
            command=self.start_scan_thread,
            fg_color="#dbeafe",
            text_color="#008f39",
            hover_color="#bfdbfe",
            border_width=1,
            border_color="#93c5fd",

        )
        self.scan_button.pack(side="left")

        self.classify_button = ctk.CTkButton(
            top_frame,
            text="Clasificar",
            image=self.ctk_icons["categoria"],
            height=42,
            corner_radius=8,
            font=("Segoe UI", 14),
            command=self.start_classification,
            fg_color="#dbeafe",
            text_color="#1e3a8a",
            hover_color="#bfdbfe",
            border_width=1,
            border_color="#93c5fd",
        )
        self.classify_button.pack(side="left", padx=(10, 0))

        progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=(12, 0))

        self.progress_bar = ctk.CTkProgressBar(progress_frame, height=14)
        self.progress_bar.pack(fill="x", expand=True, side="left")
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="0%",
            font=("Segoe UI", 12),
            width=50,
        )
        self.progress_label.pack(side="left", padx=(12, 0))

        table_outer = ctk.CTkFrame(
            self,
            fg_color="#ffffff",
            corner_radius=10,
            border_width=1,
            border_color="#d9d9d9",
        )
        table_outer.pack(fill="both", expand=True, padx=20, pady=16)

        table_container = tk.Frame(table_outer, bg="#f4f4f4")
        table_container.pack(fill="both", expand=True, padx=1, pady=1)

        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "Treeview",
            background="#ffffff",
            fieldbackground="#ffffff",
            foreground="#1a1a1a",
            rowheight=36,
            borderwidth=0,
            font=("Segoe UI", 11),
        )
        style.configure(
            "Treeview.Heading",
            background="#f6f6f6",
            foreground="#1a1a1a",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padding=8,
        )
        style.map("Treeview", background=[("selected", "#4f7294")])
        style.map("Treeview.Heading", background=[("active", "#ececec")])

        self.tree = ttk.Treeview(
            table_container,
            columns=("type", "extension", "size", "modified", "full_path"),
            show="tree headings",
        )

        self.tree.heading("#0", text="Nombre")
        self.tree.heading("type", text="Tipo")
        self.tree.heading("extension", text="Extensión")
        self.tree.heading("size", text="Tamaño")
        self.tree.heading("modified", text="Fecha modificación")
        self.tree.heading("full_path", text="Ruta completa")

        self.tree.column("#0", width=320, anchor="w")
        self.tree.column("type", width=120, anchor="center")
        self.tree.column("extension", width=120, anchor="center")
        self.tree.column("size", width=130, anchor="center")
        self.tree.column("modified", width=220, anchor="center")
        self.tree.column("full_path", width=760, anchor="w")

        y_scroll = ttk.Scrollbar(table_container, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_container, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        table_container.grid_rowconfigure(0, weight=1)
        table_container.grid_columnconfigure(0, weight=1)

        summary_frame = ctk.CTkFrame(
            self,
            fg_color="#ffffff",
            corner_radius=10,
            border_width=1,
            border_color="#d9d9d9",
        )
        summary_frame.pack(fill="x", padx=20, pady=(0, 16))

        ctk.CTkLabel(
            summary_frame,
            text="Resumen del escaneo",
            font=("Segoe UI", 16, "bold"),
            text_color="#0d5dbc",
        ).grid(row=0, column=0, columnspan=5, sticky="w", padx=20, pady=(16, 10))

        self.summary_values = {}

        self.summary_values["files"] = self._create_summary_card(
            summary_frame, 1, 0, self.ctk_icons["file"], "Archivos encontrados", "0"
        )
        self.summary_values["word"] = self._create_summary_card(
            summary_frame, 1, 1, self.ctk_icons["word"], "Word", "0"
        )

        self.summary_values["pdf"] = self._create_summary_card(
            summary_frame, 1, 2, self.ctk_icons["pdf"], "PDF", "0"
        )
        self.summary_values["images"] = self._create_summary_card(
            summary_frame, 1, 3, self.ctk_icons["image"], "Imágenes", "0"
        )
        self.summary_values["excel"] = self._create_summary_card(
            summary_frame, 1, 4, self.ctk_icons["excel"], "Excel", "0"
        )
        self.summary_values["last_scan"] = self._create_summary_card(
            summary_frame, 1, 5, self.ctk_icons["clock"], "Último escaneo", "-"
        )

        for i in range(5):
            summary_frame.grid_columnconfigure(i, weight=1)

        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=20, pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            status_frame,
            text="🟢 Listo",
            font=("Segoe UI", 13),
            text_color="#1f1f1f",
        )
        self.status_label.pack(side="left")

        self.elapsed_label = ctk.CTkLabel(
            status_frame,
            text="Tiempo total: 00:00:00",
            font=("Segoe UI", 13, "bold"),
            text_color="#1f1f1f",
        )
        self.elapsed_label.pack(side="right")

        # Overlay de procesamiento
        self.loading_overlay = tk.Frame(
            table_container,
            bg="#ffffff",
        )

        self.loading_label = tk.Label(
            self.loading_overlay,
            text="⏳ Procesando documentos...\nPor favor espera",
            font=("Segoe UI", 16, "bold"),
            bg="#ffffff",
            fg="#1a1a1a",
            justify="center"
        )

        self.loading_label.place(relx=0.5, rely=0.4, anchor="center")

        self.loading_progress_label = tk.Label(
            self.loading_overlay,
            text="",
            font=("Segoe UI", 12),
            bg="#ffffff",
            fg="#666666",
            justify="center"
        )
        self.loading_progress_label.place(relx=0.5, rely=0.6, anchor="center")

        # Oculto por defecto
        self.loading_overlay.place_forget()


    def show_loading(self):
        self.loading_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

    def hide_loading(self):
        self.loading_overlay.place_forget()

    def _create_summary_card(self, parent, row, column, icon, title, value):
        card = ctk.CTkFrame(
            parent,
            fg_color="#ffffff",
            corner_radius=0,
            border_width=0,
        )
        card.grid(row=row, column=column, sticky="nsew", padx=14, pady=(0, 18))

        # Contenedor horizontal
        inner = ctk.CTkFrame(
            card,
            fg_color="#ffffff",
            corner_radius=0,
            border_width=0,
        )
        inner.pack(fill="both", expand=True, padx=6, pady=6)

        # Icono a la izquierda
        icon_label = ctk.CTkLabel(inner, image=icon, text="")
        icon_label.pack(side="left", padx=(0, 20))

        # Texto a la derecha
        text_frame = ctk.CTkFrame(
            inner,
            fg_color="#ffffff",
            corner_radius=0,
            border_width=0,
        )
        text_frame.pack(side="left", fill="both", expand=True)

        title_label = ctk.CTkLabel(
            text_frame,
            text=title,
            font=("Segoe UI", 15),
            text_color="#222222",
            anchor="w",
        )
        title_label.pack(anchor="w")

        value_label = ctk.CTkLabel(
            text_frame,
            text=value,
            font=("Segoe UI", 15, "bold"),
            text_color="#111111",
            anchor="w",
        )
        value_label.pack(anchor="w", pady=(2, 0))

        return value_label
    def select_folder(self) -> None:
        folder = filedialog.askdirectory(title="Selecciona la carpeta raíz")
        if folder:
            self.selected_folder = folder
            self.folder_entry.delete(0, "end")
            self.folder_entry.insert(0, folder)

    def start_scan_thread(self) -> None:
        if not self.selected_folder:
            messagebox.showwarning("Aviso", "Primero selecciona una carpeta.")
            return
        if not Path(self.selected_folder).exists():
            messagebox.showerror("Error", "La carpeta no existe.")
            return

        self.scan_button.configure(state="disabled")
        self.select_button.configure(state="disabled")
        self.progress_bar.set(0)
        self.progress_label.configure(text="0%")
        self.status_label.configure(text="⏳ Escaneando...")

        threading.Thread(target=self.run_scan, daemon=True).start()

    def run_scan(self) -> None:
        folder = Path(self.selected_folder)
        if not folder.exists() or not folder.is_dir():
            self.after(0, lambda: messagebox.showerror("Error", "La ruta seleccionada no es válida."))
            self.after(0, self._restore_buttons)
            return

        self.after(0, self.clear_tree)

        start = time.perf_counter()

        def progress_callback(value: float):
            self.after(0, lambda: self._update_progress(value))

        results = scan_folder(folder, progress_callback=progress_callback)
        elapsed_seconds = time.perf_counter() - start

        self.after(0, lambda: self.populate_tree(results))
        self.after(0, lambda: self.update_summary(results))
        self.after(0, lambda: self.update_status(elapsed_seconds))
        self.after(0, self._restore_buttons)



    def _restore_buttons(self) -> None:
        self.scan_button.configure(state="normal")
        self.select_button.configure(state="normal")

    def _update_progress(self, value: float) -> None:
        self.progress_bar.set(value)
        self.progress_label.configure(text=f"{int(value * 100)}%")

    def clear_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    def populate_tree(self, results: list[dict[str, str]]) -> None:
        for row in results:
            icon = self.tree_images.get(row["type"], self.tree_images["default"])
            self.tree.insert(
                "",
                "end",
                text=row["name"],
                image=icon,
                values=(
                    row["type"],
                    row["extension"],
                    row["size"],
                    row["modified"],
                    row["full_path"],
                ),
            )

    def update_summary(self, results: list[dict[str, str]]) -> None:
        counts = Counter(row["type"] for row in results)

        self.summary_values["files"].configure(text=str(len(results)))
        self.summary_values["pdf"].configure(text=str(counts.get("PDF", 0)))
        self.summary_values["word"].configure(text=str(counts.get("Word", 0)))
        self.summary_values["images"].configure(text=str(counts.get("Imagen", 0)))
        self.summary_values["excel"].configure(text=str(counts.get("Excel", 0)))
        self.summary_values["last_scan"].configure(text=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    def update_status(self, elapsed_seconds: float) -> None:
        hh = int(elapsed_seconds // 3600)
        mm = int((elapsed_seconds % 3600) // 60)
        ss = int(elapsed_seconds % 60)

        self.status_label.configure(text="✅ Escaneo completado correctamente.")
        self.elapsed_label.configure(text=f"Tiempo total: {hh:02d}:{mm:02d}:{ss:02d}")

    def start_classification(self):
        if not self.tree.get_children():
            messagebox.showwarning("Aviso", "Primero debes escanear archivos.")
            return

        self.status_label.configure(text="🧠 Enviando archivos al backend...")
        self.show_loading()
        threading.Thread(target=self.run_classification, daemon=True).start()


    def get_files_from_tree(self):
        files = []

        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            files.append({
                "name": self.tree.item(item, "text"),
                "type": values[0],
                "extension": values[1],
                "size": values[2],
                "modified": values[3],
                "full_path": values[4],
            })

        return files


    def run_classification(self):
        files = self.get_files_from_tree()
        total = len(files)

        def progress_callback(current: int, total_files: int, file_name: str, status: str):
            """Actualiza el overlay con el documento actual (llamado desde hilo worker)."""
            if status == "extrayendo":
                msg = f"Fase 1/2 (Extracción) - {current}/{total_files}:\n{file_name}"
            else:
                estado_texto = "Analizando Semántica" if status == "procesando" else "Completado"
                msg = f"Fase 2/2 ({estado_texto}) - {current}/{total_files}:\n{file_name}"
            
            def update_ui():
                self.loading_progress_label.configure(text=msg)
                if status == "completado":
                    pct = int((current / total_files) * 100) if total_files else 100
                    self._update_progress(pct / 100.0)
                self.update_idletasks() # Fuerza a la interfaz a redibujarse INMEDIATAMENTE
            
            self.after(0, update_ui)

        results = process_documents(files, progress_callback=progress_callback)
        self.after(0, lambda: self.on_classification_done(results))

    # Mapeo de file_type (minúscula) → clave de tree_images
    _FILE_TYPE_ICON = {
        "word":  "Word",
        "pdf":   "PDF",
        "excel": "Excel",
        "image": "Imagen",
        "xml":   "XML",
        "txt":   "TXT",
    }

    def on_classification_done(self, results):
        self.hide_loading()
        self.clear_tree()

        # ── Guardar JSON completo en disco ──────────────────────────────
        try:
            ruta = "./datos.json"
            with open(ruta, "w", encoding="utf-8") as archivo:
                json.dump(results, archivo, ensure_ascii=False, indent=4)
            print(f"Archivo guardado correctamente en: {ruta}")
        except Exception as e:
            print(f"Error al guardar el archivo: {e}")

        # ── Reconfigurar columnas para vista de clasificación ───────────
        self.tree["columns"] = (
            "clasificacion",
            "confianza",
            "subtipo",
            "tema",
            "resumen",
            "full_path",
        )

        self.tree.heading("#0",            text="Documento")
        self.tree.heading("clasificacion", text="Categoría")
        self.tree.heading("confianza",     text="Confianza")
        self.tree.heading("subtipo",       text="Subtipo")
        self.tree.heading("tema",          text="Tema")
        self.tree.heading("resumen",       text="Resumen")
        self.tree.heading("full_path",     text="Ruta completa")

        self.tree.column("#0",            width=260, anchor="w")
        self.tree.column("clasificacion", width=160, anchor="center")
        self.tree.column("confianza",     width=100, anchor="center")
        self.tree.column("subtipo",       width=200, anchor="w")
        self.tree.column("tema",          width=280, anchor="w")
        self.tree.column("resumen",       width=460, anchor="w")
        self.tree.column("full_path",     width=500, anchor="w")

        # ── Poblar tabla con resultados del agente ──────────────────────
        for row in results:
            icon_key = self._FILE_TYPE_ICON.get(row.get("file_type", ""), "default")
            icon     = self.tree_images.get(icon_key, self.tree_images["default"])

            decision = row.get("agent_decision", {})
            profile  = row.get("semantic_profile", {})

            resumen = profile.get("resumen", "") or ""
            if len(resumen) > 160:
                resumen = resumen[:160] + "..."

            self.tree.insert(
                "",
                "end",
                text=row.get("file_name", ""),
                image=icon,
                values=(
                    decision.get("categoria_final", "—"),
                    decision.get("confianza",        "—"),
                    profile.get("subcategoria",      "—"),
                    profile.get("tema",              "—"),
                    resumen or "—",
                    row.get("full_path", ""),
                ),
            )

        self.status_label.configure(text="✅ Clasificación completada")

if __name__ == "__main__":
    app = DocumentIngestionApp()
    app.mainloop()
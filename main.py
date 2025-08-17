#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet académique - Démo de sensibilité d'une détection faciale basique
Style: trhacknon (dark + néons)
Auteur: toi
Usage: pédagogique uniquement

Lance:
    python trhacknon_face_demo.py
"""

import os
import math
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import Tuple, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

# ===========================
# Thème et utilitaires UI
# ===========================
TRK_BG = "#0c0f14"        # fond sombre
TRK_PANEL = "#11161f"     # panneau
TRK_NEON = "#39ff14"      # vert néon
TRK_NEON2 = "#00bfff"     # cyan néon
TRK_PINK = "#ff1493"      # magenta néon
TRK_TEXT = "#cfe3ff"      # texte clair

def style_trhacknon(root: tk.Tk):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=TRK_BG, foreground=TRK_TEXT)
    style.configure("TLabel", background=TRK_BG, foreground=TRK_TEXT)
    style.configure("TLabelframe", background=TRK_PANEL, foreground=TRK_NEON)
    style.configure("TLabelframe.Label", background=TRK_PANEL, foreground=TRK_NEON)
    style.configure("TFrame", background=TRK_PANEL)
    style.configure("TButton", background=TRK_PANEL, foreground=TRK_NEON2, padding=6, relief="flat")
    style.map("TButton",
              background=[("active", "#16212f")],
              foreground=[("active", TRK_NEON)])
    style.configure("TCombobox", fieldbackground=TRK_PANEL, background=TRK_PANEL, foreground=TRK_TEXT)
    style.map("TCombobox",
              fieldbackground=[("readonly", TRK_PANEL)],
              foreground=[("readonly", TRK_TEXT)])
    style.configure("Horizontal.TScale", background=TRK_PANEL)
    root.configure(bg=TRK_BG)

def neon_title(lbl: ttk.Label, color=TRK_NEON):
    lbl.configure(font=("Segoe UI", 16, "bold"))
    lbl.configure(foreground=color)

# ===========================
# Génération de motifs
# ===========================

def checker(size=200, cell=20, c0=(0,0,0), c1=(255,255,255)) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    a = (xx // max(2, cell)) % 2
    b = (yy // max(2, cell)) % 2
    pat = (a ^ b).astype(np.uint8)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[pat == 0] = c0
    img[pat == 1] = c1
    return Image.fromarray(img, "RGB")

def stripes(size=200, period=20, angle=30, c0=(0,255,0), c1=(255,0,255)) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    theta = math.radians(angle % 180)
    u = (math.cos(theta) * xx + math.sin(theta) * yy)
    bands = ((u // max(2, period)) % 2).astype(np.uint8)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[bands == 0] = c0
    img[bands == 1] = c1
    return Image.fromarray(img, "RGB")

def noise_rgb(size=200) -> Image.Image:
    arr = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")

def triangles(size=200, cells=12, palette=((10,10,10),(240,240,240),(255,0,0),(0,191,255),(57,255,20))) -> Image.Image:
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    cell = max(2, size // max(2, cells))
    for iy in range(0, size, cell):
        for ix in range(0, size, cell):
            x0, y0 = ix, iy
            x1, y1 = min(ix+cell, size), min(iy+cell, size)
            if ((ix // cell) + (iy // cell)) % 2 == 0:
                tri1 = [(x0,y0),(x1,y0),(x0,y1)]
                tri2 = [(x1,y0),(x1,y1),(x0,y1)]
            else:
                tri1 = [(x0,y0),(x1,y1),(x0,y1)]
                tri2 = [(x0,y0),(x1,y0),(x1,y1)]
            draw.polygon(tri1, fill=random.choice(palette))
            draw.polygon(tri2, fill=random.choice(palette))
    return img

def value_noise(size=200, tile=64, octaves=4, contrast=1.0,
                palette=((0,0,0),(57,255,20),(0,191,255),(255,20,147))) -> Image.Image:
    h = w = size
    rng = np.random.default_rng(12345)
    g = rng.random((tile+1, tile+1), dtype=np.float32)
    g[tile, :] = g[0, :]
    g[:, tile] = g[:, 0]

    yy, xx = np.mgrid[0:h, 0:w]
    u = (xx / w) * tile
    v = (yy / h) * tile

    def fade(t): return t*t*(3-2*t)

    img = np.zeros((h, w), dtype=np.float32)
    for o in range(octaves):
        f = 2**o
        uu = u * f
        vv = v * f
        x0 = np.floor(uu).astype(int) % tile
        y0 = np.floor(vv).astype(int) % tile
        x1 = (x0 + 1) % tile
        y1 = (y0 + 1) % tile
        tx = fade(uu - x0)
        ty = fade(vv - y0)
        a = g[y0, x0]
        b = g[y0, x1]
        c = g[y1, x0]
        d = g[y1, x1]
        lerp_x1 = a*(1-tx) + b*tx
        lerp_x2 = c*(1-tx) + d*tx
        val = lerp_x1*(1-ty) + lerp_x2*ty
        img += val / (2**o)

    img -= img.min()
    img /= (img.max() + 1e-8)
    img = np.clip((img - 0.5)*contrast + 0.5, 0, 1)

    steps = np.linspace(0, 1, len(palette)+1)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, col in enumerate(palette):
        mask = (img >= steps[i]) & (img < steps[i+1])
        out[mask] = col
    return Image.fromarray(out, "RGB")

# ===========================
# Détection & Patch
# ===========================
@dataclass
class DetectParams:
    scale: float = 1.1
    neighbors: int = 5

class FaceDemoApp:
    def __init__(self, root: tk.Tk):
        style_trhacknon(root)
        root.title("trhacknon — Démo faciale académique")
        root.minsize(1080, 680)
        self.root = root

        # état
        self.cv_img: Optional[np.ndarray] = None          # BGR original
        self.cv_mod: Optional[np.ndarray] = None          # BGR modifié
        self.history: List[np.ndarray] = []               # Undo
        self.faces_before = []
        self.faces_after = []
        self.det_params = DetectParams()
        self.patch_drag = {"dragging": False, "x": 0, "y": 0, "w": 200, "h": 200}
        self.patch_last = None  # dernière Image patch (RGBA)

        # Layout principal
        topbar = ttk.Frame(root)
        topbar.pack(fill="x", padx=10, pady=6)

        title = ttk.Label(topbar, text="trhacknon — Facial Sensitivity Demo")
        neon_title(title, TRK_NEON)
        title.pack(side="left")

        btn_load = ttk.Button(topbar, text="Charger image", command=self.load_image)
        btn_load.pack(side="left", padx=8)
        ttk.Button(topbar, text="Sauver original", command=lambda: self.save_current(True)).pack(side="left", padx=6)
        ttk.Button(topbar, text="Sauver modifié", command=lambda: self.save_current(False)).pack(side="left", padx=6)
        ttk.Button(topbar, text="Reset", command=self.reset_image).pack(side="left", padx=6)
        ttk.Button(topbar, text="Undo", command=self.undo).pack(side="left", padx=6)

        # Panneau latéral
        side = ttk.Labelframe(root, text="Paramètres", padding=10)
        side.pack(side="right", fill="y", padx=10, pady=6)

        # Détection
        det_fr = ttk.Labelframe(side, text="Détection HaarCascade", padding=10)
        det_fr.pack(fill="x", pady=6)
        ttk.Label(det_fr, text="scaleFactor").pack(anchor="w")
        self.scale_var = tk.DoubleVar(value=self.det_params.scale)
        ttk.Scale(det_fr, from_=1.05, to=1.5, variable=self.scale_var, orient="horizontal", command=self._update_scale).pack(fill="x")
        ttk.Label(det_fr, text="minNeighbors").pack(anchor="w", pady=(8,0))
        self.neigh_var = tk.IntVar(value=self.det_params.neighbors)
        ttk.Scale(det_fr, from_=1, to=10, variable=self.neigh_var, orient="horizontal", command=self._update_neigh).pack(fill="x")
        ttk.Button(det_fr, text="(Re)détecter", command=self.detect_both).pack(fill="x", pady=6)

        # Motif
        pat_fr = ttk.Labelframe(side, text="Motif (patch)", padding=10)
        pat_fr.pack(fill="x", pady=6)
        ttk.Label(pat_fr, text="Type").pack(anchor="w")
        self.pattern_var = tk.StringVar(value="checker")
        pat_box = ttk.Combobox(pat_fr, values=["checker","stripes","noise","triangles","value-noise"],
                               textvariable=self.pattern_var, state="readonly")
        pat_box.pack(fill="x")

        ttk.Label(pat_fr, text="Taille du patch (px)").pack(anchor="w", pady=(8,0))
        self.size_var = tk.IntVar(value=220)
        ttk.Scale(pat_fr, from_=60, to=420, variable=self.size_var, orient="horizontal").pack(fill="x")

        ttk.Label(pat_fr, text="Opacité (%)").pack(anchor="w", pady=(8,0))
        self.alpha_var = tk.IntVar(value=90)
        ttk.Scale(pat_fr, from_=10, to=100, variable=self.alpha_var, orient="horizontal").pack(fill="x")

        ttk.Label(pat_fr, text="Cellule/Période").pack(anchor="w", pady=(8,0))
        self.cell_var = tk.IntVar(value=22)
        ttk.Scale(pat_fr, from_=4, to=80, variable=self.cell_var, orient="horizontal").pack(fill="x")

        ttk.Label(pat_fr, text="Angle (stripes)").pack(anchor="w", pady=(8,0))
        self.angle_var = tk.IntVar(value=33)
        ttk.Scale(pat_fr, from_=0, to=90, variable=self.angle_var, orient="horizontal").pack(fill="x")

        ttk.Label(pat_fr, text="Densité/Octaves (value-noise)").pack(anchor="w", pady=(8,0))
        self.oct_var = tk.IntVar(value=4)
        ttk.Scale(pat_fr, from_=1, to=6, variable=self.oct_var, orient="horizontal").pack(fill="x")

        ttk.Button(pat_fr, text="Générer & placer (auto front)", command=self.apply_patch_auto).pack(fill="x", pady=(10,2))
        ttk.Button(pat_fr, text="Générer puis placer à la souris", command=self.prepare_patch_drag).pack(fill="x")

        # Stats
        st_fr = ttk.Labelframe(side, text="Résultats", padding=10)
        st_fr.pack(fill="x", pady=6)
        self.lbl_before = ttk.Label(st_fr, text="Avant: - visage(s)")
        self.lbl_before.pack(anchor="w")
        self.lbl_after = ttk.Label(st_fr, text="Après: - visage(s)")
        self.lbl_after.pack(anchor="w")

        # Zone image (avant & après, côte à côte)
        center = ttk.Frame(root)
        center.pack(fill="both", expand=True, padx=10, pady=6)
        self.canvas_before = tk.Label(center, bg=TRK_BG)
        self.canvas_after  = tk.Label(center, bg=TRK_BG)
        self.canvas_before.pack(side="left", expand=True, fill="both", padx=6)
        self.canvas_after.pack(side="left", expand=True, fill="both", padx=6)

        # Canvas pour drag
        self.canvas_after.bind("<ButtonPress-1>", self.on_press)
        self.canvas_after.bind("<B1-Motion>", self.on_drag)
        self.canvas_after.bind("<ButtonRelease-1>", self.on_release)

        # Charge le classifieur
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Splash
        self.draw_splash()

    # ---------- UI Helpers ----------
    def draw_splash(self):
        splash = Image.new("RGB", (1024, 576), (12, 15, 20))
        d = ImageDraw.Draw(splash)
        # cadre néon
        d.rectangle([10,10,1014,566], outline=(57,255,20), width=3)
        msg = [
            "trhacknon — Démo académique",
            "• Charger une image (portrait libre).",
            "• Cliquer 'Générer & placer (auto front)' OU 'placer à la souris'.",
            "• (Re)détecter pour comparer Avant/Après.",
            "• Ajuster taille / opacité / paramètres de motif.",
            "• Undo / Reset / Sauvegarder.",
        ]
        y = 60
        for i, line in enumerate(msg):
            color = (57,255,20) if i in (0,) else (207,227,255)
            d.text((40,y), line, fill=color)
            y += 48
        tkimg = ImageTk.PhotoImage(splash)
        self.canvas_before.configure(image=tkimg)
        self.canvas_before.image = tkimg
        self.canvas_after.configure(image=tkimg)
        self.canvas_after.image = tkimg

    def _update_scale(self, _=None):
        self.det_params.scale = float(self.scale_var.get())

    def _update_neigh(self, _=None):
        self.det_params.neighbors = int(self.neigh_var.get())

    def imshow_pair(self):
        """Affiche original et modifié avec rectangles de détection."""
        if self.cv_img is None:
            return
        img_disp = self.draw_faces(self.cv_img.copy(), self.faces_before, color=(0,191,255))
        mod_disp = (self.cv_mod if self.cv_mod is not None else self.cv_img).copy()
        mod_disp = self.draw_faces(mod_disp, self.faces_after, color=(255,20,147))
        self._set_label_image(self.canvas_before, img_disp)
        self._set_label_image(self.canvas_after, mod_disp)

    def _set_label_image(self, label: tk.Label, bgr_img: np.ndarray):
        # fit to label size (auto); keep aspect
        h, w = bgr_img.shape[:2]
        # target width ~ 512 each (responsive simple)
        target_w = max(480, min(720, w))
        scale = target_w / float(w)
        target_h = int(h * scale)
        rgb = cv2.cvtColor(cv2.resize(bgr_img, (target_w, target_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        tkimg = ImageTk.PhotoImage(pil)
        label.configure(image=tkimg)
        label.image = tkimg

    @staticmethod
    def draw_faces(bgr: np.ndarray, faces, color=(0,191,255)):
        for (x,y,w,h) in faces:
            cv2.rectangle(bgr, (x,y), (x+w,y+h), color, 2)
        return bgr

    # ---------- Actions ----------
    def load_image(self):
        path = filedialog.askopenfilename(title="Choisir une image",
                                          filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erreur", "Impossible de charger l'image.")
            return
        self.cv_img = img
        self.cv_mod = img.copy()
        self.history.clear()
        self.detect_both()
        self.imshow_pair()

    def save_current(self, original=True):
        if self.cv_img is None:
            return
        arr = self.cv_img if original else (self.cv_mod if self.cv_mod is not None else self.cv_img)
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            initialfile=("original.png" if original else "modifie.png"))
        if path:
            cv2.imwrite(path, arr)
            messagebox.showinfo("OK", f"Image sauvegardée:\n{path}")

    def reset_image(self):
        if self.cv_img is None:
            return
        self.cv_mod = self.cv_img.copy()
        self.history.clear()
        self.detect_both()
        self.imshow_pair()

    def undo(self):
        if not self.history:
            return
        self.cv_mod = self.history.pop(-1)
        self.detect_both()
        self.imshow_pair()

    def detect_both(self):
        if self.cv_img is None:
            return
        self.faces_before = self.cascade.detectMultiScale(self.cv_img, self.det_params.scale, self.det_params.neighbors)
        target = self.cv_mod if self.cv_mod is not None else self.cv_img
        self.faces_after = self.cascade.detectMultiScale(target, self.det_params.scale, self.det_params.neighbors)
        self.lbl_before.configure(text=f"Avant: {len(self.faces_before)} visage(s)")
        self.lbl_after.configure(text=f"Après: {len(self.faces_after)} visage(s)")
        self.imshow_pair()

    # ---------- Patch generation & placement ----------
    def build_patch(self) -> Image.Image:
        size = int(self.size_var.get())
        typ = self.pattern_var.get()
        alpha = int(self.alpha_var.get())
        cell = int(self.cell_var.get())
        angle = int(self.angle_var.get())
        octv = int(self.oct_var.get())

        if   typ == "checker":
            im = checker(size=size, cell=cell)
        elif typ == "stripes":
            im = stripes(size=size, period=cell, angle=angle)
        elif typ == "noise":
            im = noise_rgb(size=size)
        elif typ == "triangles":
            im = triangles(size=size, cells=max(2, cell//2))
        elif typ == "value-noise":
            im = value_noise(size=size, tile=max(8, cell), octaves=octv, contrast=1.0)
        else:
            im = checker(size=size, cell=cell)

        # alpha
        im = im.convert("RGBA")
        a = Image.new("L", im.size, int(255 * (alpha/100.0)))
        im.putalpha(a)
        return im

    def apply_patch_auto(self):
        if self.cv_img is None:
            return
        patch = self.build_patch()
        self.patch_last = patch.copy()

        # where: sur le front du 1er visage détecté
        faces = self.faces_before if len(self.faces_before) else \
                self.cascade.detectMultiScale(self.cv_mod, self.det_params.scale, self.det_params.neighbors)
        if len(faces) == 0:
            messagebox.showwarning("Info", "Aucun visage détecté. Place à la souris possible.")
            return

        x,y,w,h = faces[0]
        px = int(x + w*0.25)
        py = int(y + h*0.05)  # haut/front
        psize = patch.size[0]
        # historique pour Undo
        self.history.append(self.cv_mod.copy())
        self.cv_mod = self.paste_rgba(self.cv_mod, patch, (px, py))
        self.detect_both()
        self.imshow_pair()

    def prepare_patch_drag(self):
        if self.cv_img is None:
            return
        patch = self.build_patch()
        self.patch_last = patch.copy()
        self.patch_drag["w"], self.patch_drag["h"] = patch.size
        messagebox.showinfo("Placement", "Clique & glisse sur l'image de droite pour positionner le patch.")

    def on_press(self, event):
        if self.cv_img is None or self.patch_last is None:
            return
        self.patch_drag["dragging"] = True
        self.patch_drag["x"] = event.x
        self.patch_drag["y"] = event.y

    def on_drag(self, event):
        if not self.patch_drag["dragging"] or self.cv_img is None or self.patch_last is None:
            return
        # feedback visuel optionnel (on pourrait dessiner un rectangle fantôme)
        # ici on attend le release pour appliquer

    def on_release(self, event):
        if not self.patch_drag["dragging"] or self.cv_img is None or self.patch_last is None:
            return
        self.patch_drag["dragging"] = False

        # Convertir coords label -> coords image modifiée
        # On récupère la taille actuellement affichée dans canvas_after.image
        tkimg = getattr(self.canvas_after, "image", None)
        if tkimg is None:
            return
        disp_w = tkimg.width()
        disp_h = tkimg.height()

        h, w = (self.cv_mod if self.cv_mod is not None else self.cv_img).shape[:2]
        # On suppose un fit par largeur (voir _set_label_image)
        # Calcule l'échelle réelle
        if w == 0 or disp_w == 0:
            return
        scale = w / float(disp_w)
        dx = int(event.x * scale)
        dy = int(event.y * scale)

        self.history.append(self.cv_mod.copy())
        self.cv_mod = self.paste_rgba(self.cv_mod, self.patch_last, (dx - self.patch_last.size[0]//2,
                                                                     dy - self.patch_last.size[1]//2))
        self.detect_both()
        self.imshow_pair()

    @staticmethod
    def paste_rgba(bgr_img: np.ndarray, patch_rgba: Image.Image, topleft: Tuple[int,int]) -> np.ndarray:
        """Colle une image RGBA (PIL) sur un fond BGR (np.ndarray)."""
        x, y = topleft
        ph, pw = patch_rgba.size[1], patch_rgba.size[0]
        H, W = bgr_img.shape[:2]
        # clip
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x+pw), min(H, y+ph)
        if x0 >= x1 or y0 >= y1:
            return bgr_img
        crop = patch_rgba.crop((x0-x, y0-y, x1-x, y1-y))
        # convert to arrays
        patch = np.array(crop)  # RGBA
        alpha = (patch[:, :, 3:4] / 255.0)
        patch_rgb = patch[:, :, :3][:, :, ::-1]  # RGB->BGR
        bg = bgr_img[y0:y1, x0:x1].astype(np.float32)
        fg = patch_rgb.astype(np.float32)
        out = fg * alpha + bg * (1 - alpha)
        bgr_img[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
        return bgr_img

# ===========================
# main
# ===========================
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDemoApp(root)
    root.mainloop()

import tkinter as tk
from tkinter import messagebox
import numpy as np

#MLP + MNIST 
from mlp_mnist import MLP, load_mnist_local

# Segédfüggvények model mentéshez / betöltéshez

def save_model(model, filename="model.npz"):
    data = {}
    for i, layer in enumerate(model.layers):
        data[f"layer{i}_W"] = layer.W
        data[f"layer{i}_b"] = layer.b
    np.savez(filename, **data)
    print(f"Model mentve: {filename}")


def load_model(filename="model.npz"):
    try:
        npz = np.load(filename)
    except FileNotFoundError:
        messagebox.showerror("Hiba", f"A(z) {filename} nem található.")
        return None

    # réteg méretek visszafejtése a W mátrixokból
    layer_sizes = []
    i = 0
    while f"layer{i}_W" in npz:
        W = npz[f"layer{i}_W"]
        if i == 0:
            layer_sizes.append(W.shape[0])   # input_dim
        layer_sizes.append(W.shape[1])       # output_dim
        i += 1

    if len(layer_sizes) < 2:
        messagebox.showerror("Hiba", "Érvénytelen modell fájl.")
        return None

    # aktivációk: ReLU az összes rejtett rétegen, Softmax a végén
    activations = ["relu"] * (len(layer_sizes) - 2) + ["softmax"]

    model = MLP(layer_sizes, activations)

    for i, layer in enumerate(model.layers):
        layer.W = npz[f"layer{i}_W"]
        layer.b = npz[f"layer{i}_b"]

    print(f"Model betöltve: {filename}")
    return model

# # Szín-mappelő segédek

def activation_to_color(a, min_a, max_a):
    # ReLU eset: többnyire >= 0
    if max_a > min_a:
        v = (a - min_a) / (max_a - min_a)
    else:
        v = 0.0
    v = max(0.0, min(1.0, float(v)))
    g = int(255 * v)
    return f"#{g:02x}{g:02x}{g:02x}"  # szürke skála


def weight_to_color(w, max_abs_w):
    if max_abs_w <= 0:
        return "#808080"
    v = min(1.0, abs(float(w)) / max_abs_w)
    c = int(255 * v)
    if w >= 0:
        return f"#00{c:02x}00"  # zöld pozitív
    else:
        return f"#{c:02x}0000"  # piros negatív

# GUI osztály

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interaktív MLP MNIST + rajztábla")

        self.model = MLP(
            layer_sizes=[784, 128, 64, 10],
            activations=["relu", "relu", "softmax"]
        )

        self.learning_rate = 0.01
        self.max_neurons_to_display = 50

        self.grid_size = 28
        self.canvas_size = 560
        self.cell_size = self.canvas_size // self.grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        self._build_layout()

        self._compute_network_layout()

        self.update_prediction_and_visualization()

    # GUI 

    def _build_layout(self):
        # Bal: rajztábla + gombok
        left_frame = tk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.draw_canvas = tk.Canvas(
            left_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black"
        )
        self.draw_canvas.pack()

        self.draw_canvas.bind("<B1-Motion>", self.on_draw)
        self.draw_canvas.bind("<Button-1>", self.on_draw)

        btn_frame = tk.Frame(left_frame)
        btn_frame.pack(pady=5, fill=tk.X)

        clear_btn = tk.Button(btn_frame, text="Törlés", command=self.clear_canvas)
        clear_btn.pack(side=tk.LEFT, padx=2)

        save_btn = tk.Button(btn_frame, text="Model mentése", command=self.on_save_model)
        save_btn.pack(side=tk.LEFT, padx=2)

        load_btn = tk.Button(btn_frame, text="Model betöltése", command=self.on_load_model)
        load_btn.pack(side=tk.LEFT, padx=2)

        train_btn = tk.Button(btn_frame, text="Gyors MNIST tanítás", command=self.on_quick_train)
        train_btn.pack(side=tk.LEFT, padx=2)

        middle_frame = tk.Frame(self.root)
        middle_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        self.pred_label = tk.Label(middle_frame, text="Predikció: -", font=("Arial", 20))
        self.pred_label.pack(pady=10)

        self.probs_label = tk.Label(middle_frame, text="", font=("Consolas", 10), justify=tk.LEFT)
        self.probs_label.pack(pady=5)

        tk.Label(middle_frame, text="Ha rossz a predikció, válaszd ki a helyes számot:", font=("Arial", 10)).pack(pady=5)
        corr_frame = tk.Frame(middle_frame)
        corr_frame.pack(pady=5)

        for d in range(10):
            b = tk.Button(corr_frame, text=str(d), width=2,
                          command=lambda d=d: self.on_correct_label(d))
            b.pack(side=tk.LEFT, padx=1)

        # Jobb: háló vizualizáció
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.LEFT, padx=10, pady=10)

        tk.Label(right_frame, text=f"Háló vizualizáció (max {self.max_neurons_to_display} neuron/réteg)", font=("Arial", 12)).pack()

        self.network_canvas_width = 600
        self.network_canvas_height = 600
        self.network_canvas = tk.Canvas(
            right_frame,
            width=self.network_canvas_width,
            height=self.network_canvas_height,
            bg="white"
        )
        self.network_canvas.pack(pady=5)

    # Rajztábla

    def on_draw(self, event):
        x, y = event.x, event.y
        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            j = x // self.cell_size  
            i = y // self.cell_size  
            if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                # cella "bekapcsolása"
                self.grid[i, j] = 1.0
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.draw_canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="white")

        self.update_prediction_and_visualization()

    def clear_canvas(self):
        self.draw_canvas.delete("all")
        self.grid[:] = 0.0
        self.update_prediction_and_visualization()

    # Modell műveletek 

    def on_save_model(self):
        save_model(self.model, "model.npz")
        messagebox.showinfo("Mentés", "Model mentve: model.npz")

    def on_load_model(self):
        m = load_model("model.npz")
        if m is not None:
            self.model = m
            self._compute_network_layout()
            self.update_prediction_and_visualization()
            messagebox.showinfo("Betöltés", "Model betöltve: model.npz")

    def on_quick_train(self):
        answer = messagebox.askyesno(
            "Gyors tanítás",
            "Ez pár másodpercig tarthat.\n\n"
            "Tanítsuk az MLP-t egy kisebb MNIST részhalmazon?"
        )
        if not answer:
            return

        try:
            (x_train, y_train), _ = load_mnist_local()
        except Exception as e:
            messagebox.showerror("Hiba", f"MNIST betöltési hiba: {e}")
            return

        # Gyorsabb próba kedvéért használjunk kevesebb adatot + keverjük
        perm = np.random.permutation(len(x_train))

        x_train_small = x_train[perm][:32000]
        y_train_small = y_train[perm][:32000]

        self.model.fit(
            x_train_small,
            y_train_small,
            epochs=5,
            batch_size=64,
            lr=self.learning_rate,
            verbose=True
        )

        self.update_prediction_and_visualization()
        messagebox.showinfo("Tanítás kész", "Gyors MNIST tanítás befejezve (5 epoch, 32k minta).")


    def on_correct_label(self, correct_digit):
        # tanítás a jelenlegi rajzból
        x = self.grid.flatten().reshape(1, -1)
        y = np.array([correct_digit], dtype=np.int64)

        loss = self.model.fit_batch(x, y, self.learning_rate)
        print(f"Online tanítás: helyes={correct_digit}, loss={loss:.4f}")

        self.update_prediction_and_visualization()

    # Háló elrendezés és rajzolás 

    def _compute_network_layout(self):
        """
        Előre kiszámoljuk, hogy a neuronok hol legyenek a network_canvas-on.
        Minden réteghez elmentjük: milyen indexű neuronokat mutatunk, és azok (x,y) pozícióit.
        """
        self.layer_indices = []    # per réteg neuronindexek (eredeti indexek)
        self.node_positions = []   # per réteg (x,y) koordináták

        num_layers = len(self.model.layers) + 1  # input + minden DenseLayer
        width = self.network_canvas_width
        height = self.network_canvas_height

        # X pozíciók rétegenként
        layer_x_positions = []
        for l in range(num_layers):
            x = (l + 1) * width / (num_layers + 1)
            layer_x_positions.append(x)

        # INPUT réteg
        input_size = self.model.layers[0].input_dim
        n_in_display = min(self.max_neurons_to_display, input_size)
        idx_in = np.linspace(0, input_size - 1, n_in_display, dtype=int)
        self.layer_indices.append(idx_in)

        y_positions_input = np.linspace(50, height - 50, n_in_display)
        self.node_positions.append(
            [(layer_x_positions[0], y) for y in y_positions_input]
        )

        # További rétegek
        for l, layer in enumerate(self.model.layers):
            size = layer.output_dim
            n_display = min(self.max_neurons_to_display, size)
            idx = np.linspace(0, size - 1, n_display, dtype=int)
            self.layer_indices.append(idx)

            y_positions = np.linspace(50, height - 50, n_display)
            self.node_positions.append(
                [(layer_x_positions[l + 1], y) for y in y_positions]
            )

    def update_prediction_and_visualization(self):
        # 1) Forward a jelenlegi rajzról
        x = self.grid.flatten().reshape(1, -1)  
        probs = self.model.forward(x)           
        probs = probs[0]                        
        pred_digit = int(np.argmax(probs))

        # 2) Label frissítés
        self.pred_label.config(text=f"Predikció: {pred_digit}")

        probs_text = "\n".join(
            f"{d}: {probs[d]:.3f}"
            for d in range(10)
        )
        self.probs_label.config(text=probs_text)

        # 3) Háló vizualizáció
        self.draw_network(x, probs)

    def draw_network(self, x_input, probs):
        self.network_canvas.delete("all")

        # Rétegenkénti aktivációk (1 mintára)
        activations = []

        # Input aktiváció
        A0 = x_input[0] 
        activations.append(A0)

        # Rejtett + kimeneti rétegek
        for layer in self.model.layers:
            A = layer.A[0] 
            activations.append(A)

        # Neuronok mérete
        radius = 6

        # Aktiváció minimum / maximum rétegenként (a színezéshez)
        layer_minmax = []
        for l, idxs in enumerate(self.layer_indices):
            a = activations[l][idxs]
            if len(a) > 0:
                mn = float(np.min(a))
                mx = float(np.max(a))
            else:
                mn, mx = 0.0, 1.0
            layer_minmax.append((mn, mx))

        # Vonalak kirajzolása (kapcsolatok)
        for l in range(len(self.layer_indices) - 1):
            # prev réteg indexei és pozíciói
            prev_idxs = self.layer_indices[l]
            prev_positions = self.node_positions[l]
            # current réteg
            curr_idxs = self.layer_indices[l + 1]
            curr_positions = self.node_positions[l + 1]

            W = self.model.layers[l].W  

            # max abs weight a kijelölt kapcsolatokra
            if len(prev_idxs) > 0 and len(curr_idxs) > 0:
                subW = W[np.ix_(prev_idxs, curr_idxs)]
                max_abs = float(np.max(np.abs(subW))) if np.any(subW) else 0.0
            else:
                max_abs = 0.0

            for i, pi in enumerate(prev_idxs):
                x1, y1 = prev_positions[i]
                for j, pj in enumerate(curr_idxs):
                    x2, y2 = curr_positions[j]
                    w = W[pi, pj]
                    color = weight_to_color(w, max_abs)
                    self.network_canvas.create_line(x1, y1, x2, y2, fill=color)

        # Neuron pontok kirajzolása
        for l, (idxs, positions) in enumerate(zip(self.layer_indices, self.node_positions)):
            mn, mx = layer_minmax[l]
            a = activations[l]
            for k, idx in enumerate(idxs):
                x, y = positions[k]
                val = a[idx]
                color = activation_to_color(val, mn, mx)
                self.network_canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color,
                    outline="black"
                )

# main

if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

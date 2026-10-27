import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,7))

# Log axes & grid
plt.xscale("log")
plt.yscale("log")
plt.grid(True, which="both", linestyle=":", linewidth=0.5)

# --------- Key competitors (manually aligned like reference) ---------

# IoT / Very-low-power
iot = {
    "Syntiant": (0.015, 80),
    "Maxim":    (0.08, 120),
}
for name, (px, py) in iot.items():
    plt.scatter(px, py, marker="v", color="deepskyblue")
    plt.text(px*1.15, py*1.15, name, fontsize=9)

# Embedded
embedded = {
    "Kendryte K210": (0.5, 400),
    "ARM Ethos-U55": (3, 5e3),
    "Jetson Nano":   (10, 3e4),
}
for name, (px, py) in embedded.items():
    plt.scatter(px, py, marker="o", color="orange")
    plt.text(px*1.07, py*1.12, name, fontsize=9)

# Datacenter chips
chips = {
    "Google TPU v3": (250, 5e5),
    "NVIDIA V100":   (300, 1e6),
}
for name, (px, py) in chips.items():
    plt.scatter(px, py, marker="^", color="green")
    plt.text(px*1.03, py*1.07, name, fontsize=9)

# Datacenter systems
systems = {
    "DGX-A100": (4000, 4e6),
    "TPU Pod":  (6000, 8e6),
}
for name,(px,py) in systems.items():
    plt.scatter(px, py, marker="s", color="limegreen")
    plt.text(px*1.02, py*1.07, name, fontsize=9)

# --------- Efficiency diagonals ---------
power = np.logspace(-2,4,400)
for tops in [1, 10, 100]:
    perf = (tops*1e3)*power
    plt.loglog(power, perf, linestyle="--", color="gray")
    plt.text(power[-1], perf[-1], f"{tops} TOPS/W", fontsize=9, ha="right")

# --------- Neuromorphic X2 Training Energy ---------
train_power = 18
train_perf  = 0.0045 * train_power * 1e3  # GOPS

plt.scatter([train_power], [train_perf], s=280, marker="*", color="red", zorder=10)
plt.annotate("Training Energy (X2 130nm)\n0.0045 TOPS/W @ 18 W",
             xy=(train_power, train_perf),
             xytext=(train_power*1.6, train_perf*2.2),
             arrowprops=dict(arrowstyle="->", lw=1.5),
             fontsize=10)

# --------- Neuromorphic X2 Inference Energy (UPDATED) ---------
inf_power = 0.113
inf_eff   = 11.05                      # TOPS/W
inf_perf  = inf_eff * inf_power * 1e3  # GOPS  -> ~1249 GOPS

plt.scatter([inf_power], [inf_perf], s=260, marker="*", color="purple", zorder=10)
plt.annotate("Inference Energy (X2 130nm)\n11.05 TOPS/W @ 0.113 W",
             xy=(inf_power, inf_perf),
             xytext=(inf_power*2.5, inf_perf*2),
             arrowprops=dict(arrowstyle="->", lw=1.5, color="purple"),
             fontsize=10, color="purple")

# --------- Labels & title ---------
plt.xlabel("Peak Power (W)")
plt.ylabel("Peak Performance (GOPS/s)")
plt.title("Neuromorphic X2 130nm: Training vs Inference Energy\nCompared Against Key Competitors")

plt.tight_layout()

out_path = r"c:\Users\KannaSridhar\Desktop\Behavioral_py\NeuromorphicX2_training_inference_star_added.png"
plt.savefig(out_path)
plt.show()

out_path


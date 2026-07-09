"""
compare.py

Orchestrates the ablation study comparing:
1. Andrej Karpathy's original classical microGPT (pure Python, CPU-only)
2. PyTorch classical microGPT (GPU-accelerated)
3. Hybrid Quantum microGPT (1 classical head on GPU, 3 quantum heads on CPU - Qiskit)
4. Hybrid Quantum microGPT (1 classical head on GPU, 3 quantum heads on GPU - PennyLane lightning.gpu)

Trains models, prints comparison table, saves loss convergence curve.
"""

import json
import matplotlib.pyplot as plt
import torch
import microgpt_classical
import classical_baseline
import microgpt_quantum
import time

# Try to import PennyLane, fall back to Qiskit if not available
try:
    import microgpt_pennylane
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False
    print("PennyLane not installed! Falling back to Qiskit quantum model.")
    print("To install PennyLane, run: pip install pennylane-lightning")
    print("For GPU support: pip install pennylane-lightning[gpu]")


def run_ablation_study(use_pytorch_classical=True, use_pennylane_quantum=True):
    """
    Args:
        use_pytorch_classical: If True, use the PyTorch classical model (GPU-accelerated).
                              If False, use the original pure Python classical model.
        use_pennylane_quantum: If True, use PennyLane version (GPU-ready) for quantum model.
                               If False, use original Qiskit version.
    """
    print("=" * 110)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 110)

    # 1. Run Classical model
    t0 = time.perf_counter()
    if use_pytorch_classical:
        print("\n--- Running PyTorch Classical microGPT ---")
        classical_res = classical_baseline.train_and_evaluate()
        classical_label = "PyTorch Classical microGPT"
        classical_color = "royalblue"
    else:
        print("\n--- Running Original Classical microGPT ---")
        classical_res = microgpt_classical.train_and_evaluate()
        classical_label = "Original Classical microGPT"
        classical_color = "slateblue"
    t_classical = time.perf_counter() - t0

    # 2. Run Hybrid Quantum microGPT
    t1 = time.perf_counter()
    if use_pennylane_quantum and PENNYLANE_AVAILABLE:
        print("\n--- Running Hybrid Quantum microGPT (PennyLane) ---")
        quantum_res = microgpt_pennylane.train_and_evaluate(use_noisy_eval=False, use_gpu=True)
        quantum_label = "Hybrid Quantum microGPT (PennyLane lightning.gpu)"
        quantum_color = "forestgreen"
    else:
        print("\n--- Running Hybrid Quantum microGPT (Qiskit) ---")
        quantum_res = microgpt_quantum.train_and_evaluate(use_noisy_eval=True)
        quantum_label = "Hybrid Quantum microGPT (Qiskit CPU)"
        quantum_color = "darkorange"
    t_quantum = time.perf_counter() - t1

    print("\n" + "=" * 120)
    print(f"                    ABLATION STUDY: {classical_label} vs {quantum_label}")
    print("=" * 120)

    q_params = quantum_res["total_params"]
    vqc_params = quantum_res["quantum_params"]
    q_init_loss = quantum_res["losses"][0]
    q_final_loss = quantum_res["losses"][-1]
    q_noisy_val_loss = quantum_res["noisy_val_loss"]
    q_samples = ", ".join(quantum_res["inference_samples"][:3]) + "..."
    q_time = quantum_res["training_time"]
    q_reduction = q_init_loss - q_final_loss

    final_step_label = f"Final Loss (step {len(quantum_res['losses'])})"

    if classical_res is not None:
        c_params = classical_res["total_params"]
        c_init_loss = classical_res["losses"][0]
        c_final_loss = classical_res["losses"][-1]
        c_reduction = c_init_loss - c_final_loss
        c_samples = ", ".join(classical_res["inference_samples"][:3]) + "..."
        c_time = classical_res["training_time"]

        print(f"{'Metric':<30} | {classical_label:<35} | {quantum_label:<35}")
        print("-" * 120)
        print(f"{'Total Parameters':<30} | {c_params:<35} | {q_params:<35}")
        print(f"{'Quantum Parameters':<30} | {'N/A':<35} | {vqc_params:<35}")
        print(f"{'Initial Loss (step 1)':<30} | {c_init_loss:<35.4f} | {q_init_loss:<35.4f}")
        print(f"{final_step_label:<30} | {c_final_loss:<35.4f} | {q_final_loss:<35.4f}")
        print(f"{'Loss Reduction':<30} | {c_reduction:<35.4f} | {q_reduction:<35.4f}")
        print(f"{'Training Time (seconds)':<30} | {c_time:<35.1f} | {q_time:<35.1f}")
        print(f"{'Inference Samples (temp=0.5)':<30} | {c_samples:<35} | {q_samples:<35}")
        print("=" * 120)

        plt.figure(figsize=(14, 7))
        plt.plot(classical_res["losses"], label=classical_label, color=classical_color, alpha=0.8, linewidth=2)
        plt.plot(quantum_res["losses"], label=quantum_label, color=quantum_color, alpha=0.8, linewidth=2)
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Cross Entropy Loss", fontsize=12)
        plt.title(f"Training Loss Comparison: {classical_label} vs {quantum_label}", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
    else:
        print(f"{'Metric':<30} | {quantum_label:<35}")
        print("-" * 70)
        print(f"{'Total Parameters':<30} | {q_params:<35}")
        print(f"{'Quantum Parameters':<30} | {vqc_params:<35}")
        print(f"{'Initial Loss (step 1)':<30} | {q_init_loss:<35.4f}")
        print(f"{final_step_label:<30} | {q_final_loss:<35.4f}")
        print(f"{'Training Time (seconds)':<30} | {q_time:<35.1f}")
        print(f"{'Inference Samples (temp=0.5)':<30} | {q_samples:<35}")
        print("=" * 70)

        plt.figure(figsize=(10, 6))
        plt.plot(quantum_res["losses"], label=quantum_label, color=quantum_color, alpha=0.8)
        plt.xlabel("Training Step")
        plt.ylabel("Cross Entropy Loss")
        plt.title(f"Training Loss: {quantum_label}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

    # Save plot
    output_plot_path = "loss_comparison.png"
    plt.savefig(output_plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    history_payload = {
        "train_loss": classical_res.get("losses", []) if classical_res is not None else quantum_res.get("losses", []),
        "val_loss": quantum_res.get("losses", []) if quantum_res is not None else classical_res.get("losses", []),
    }
    with open("training_history.json", "w") as f:
        json.dump(history_payload, f, indent=2)

    print(f"Loss comparison plot saved to {output_plot_path}")
    print("Training/validation loss history saved to training_history.json")


if __name__ == "__main__":
    # Set to True for PyTorch classical (recommended), False for original
    # Set to True for PennyLane quantum (GPU-ready), False for Qiskit
    run_ablation_study(use_pytorch_classical=True, use_pennylane_quantum=True)


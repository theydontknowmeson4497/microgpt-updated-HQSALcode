"""
compare.py

Orchestrates the ablation study comparing:
1. Andrej Karpathy's original classical microGPT (pure Python, CPU-only)
2. PyTorch classical microGPT (GPU-accelerated)
3. Hybrid Quantum microGPT (1 classical head on GPU, 3 quantum heads on CPU)

Trains models, prints comparison table, saves loss convergence curve.
"""

import matplotlib.pyplot as plt
import torch
import microgpt_classical
import classical_baseline
import microgpt_quantum
import time

def run_ablation_study(use_pytorch_classical=True):
    """
    Args:
        use_pytorch_classical: If True, use the PyTorch classical model (GPU-accelerated).
                              If False, use the original pure Python classical model.
    """
    print("=" * 90)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("=" * 90)

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
    print("\n--- Running Hybrid Quantum microGPT ---")
    quantum_res = microgpt_quantum.train_and_evaluate(use_noisy_eval=True)
    t_quantum = time.perf_counter() - t1
    
    print("\n" + "=" * 110)
    print(f"                    ABLATION STUDY: {classical_label} vs Hybrid Quantum microGPT")
    print("=" * 110)

    q_params = quantum_res["total_params"]
    vqc_params = quantum_res["quantum_params"]
    q_init_loss = quantum_res["losses"][0]
    q_final_loss = quantum_res["losses"][-1]
    q_noisy_val_loss = quantum_res["noisy_val_loss"]
    q_samples = ", ".join(quantum_res["inference_samples"][:3]) + "..."
    q_time = quantum_res["training_time"]
    q_reduction = q_init_loss - q_final_loss

    if classical_res is not None:
        c_params = classical_res["total_params"]
        c_init_loss = classical_res["losses"][0]
        c_final_loss = classical_res["losses"][-1]
        c_reduction = c_init_loss - c_final_loss
        c_samples = ", ".join(classical_res["inference_samples"][:3]) + "..."
        c_time = classical_res["training_time"]

        print(f"{'Metric':<30} | {classical_label:<30} | {'Quantum microGPT (Clean)':<25} | {'Quantum (Noisy Eval)':<22}")
        print("-" * 110)
        print(f"{'Total Parameters':<30} | {c_params:<30} | {q_params:<25} | {q_params:<22}")
        print(f"{'Quantum Parameters':<30} | {'N/A':<30} | {vqc_params:<25} | {vqc_params:<22}")
        print(f"{'Initial Loss (step 1)':<30} | {c_init_loss:<30.4f} | {q_init_loss:<25.4f} | {q_init_loss:<22.4f}")
        print(f"{'Final Loss (step 1000)':<30} | {c_final_loss:<30.4f} | {q_final_loss:<25.4f} | {q_noisy_val_loss:<22.4f}" if isinstance(q_noisy_val_loss, float) else f"{'Final Loss (step 1000)':<30} | {c_final_loss:<30.4f} | {q_final_loss:<25.4f} | {q_noisy_val_loss:<22}")
        print(f"{'Loss Reduction':<30} | {c_reduction:<30.4f} | {q_reduction:<25.4f} | {'N/A':<22}")
        print(f"{'Training Time (seconds)':<30} | {c_time:<30.1f} | {q_time:<25.1f} | {'N/A':<22}")
        print(f"{'Inference Samples (temp=0.5)':<30} | {c_samples:<30} | {q_samples:<25} | {'N/A':<22}")
        print("=" * 110)

        plt.figure(figsize=(12, 7))
        plt.plot(classical_res["losses"], label=classical_label, color=classical_color, alpha=0.8, linewidth=2)
        plt.plot(quantum_res["losses"], label="Hybrid Quantum microGPT", color="darkorange", alpha=0.8, linewidth=2)
        plt.xlabel("Training Step", fontsize=12)
        plt.ylabel("Cross Entropy Loss", fontsize=12)
        plt.title(f"Training Loss Comparison: {classical_label} vs Hybrid Quantum microGPT", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
    else:
        print(f"{'Metric':<30} | {'Quantum microGPT (Clean)':<25} | {'Quantum (Noisy Eval)':<22}")
        print("-" * 77)
        print(f"{'Total Parameters':<30} | {q_params:<25} | {q_params:<22}")
        print(f"{'Quantum Parameters':<30} | {vqc_params:<25} | {vqc_params:<22}")
        print(f"{'Initial Loss (step 1)':<30} | {q_init_loss:<25.4f} | {q_init_loss:<22.4f}")
        print(f"{'Final Loss (step 1000)':<30} | {q_final_loss:<25.4f} | {q_noisy_val_loss:<22.4f}" if isinstance(q_noisy_val_loss, float) else f"{'Final Loss (step 1000)':<30} | {q_final_loss:<25.4f} | {q_noisy_val_loss:<22}")
        print(f"{'Training Time (seconds)':<30} | {q_time:<25.1f} | {'N/A':<22}")
        print(f"{'Inference Samples (temp=0.5)':<30} | {q_samples:<25} | {'N/A':<22}")
        print("=" * 77)

        plt.figure(figsize=(10, 6))
        plt.plot(quantum_res["losses"], label="Hybrid Quantum microGPT", color="darkorange", alpha=0.8)
        plt.xlabel("Training Step")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Training Loss: Hybrid Quantum microGPT")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

    # Save the plot
    output_plot_path = "loss_comparison.png"
    plt.savefig(output_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Loss comparison plot saved to {output_plot_path}")

if __name__ == "__main__":
    # Set to True for PyTorch classical model (GPU-accelerated, recommended!)
    # Set to False for original pure Python classical model (CPU-only)
    run_ablation_study(use_pytorch_classical=True)

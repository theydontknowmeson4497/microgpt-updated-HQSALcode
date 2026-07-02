"""
compare.py

Orchestrates the ablation study comparing Andrej Karpathy's classical microGPT
against the Hybrid Quantum microGPT. Trains both models, runs noisy evaluation,
prints a comparison table, and saves the loss convergence curve as a PNG.
"""

import matplotlib.pyplot as plt
import microgpt_classical
import microgpt_quantum
import time

def run_ablation_study():
    print("=" * 70)
    # 1. Run Classical microGPT
    t0 = time.perf_counter()
    classical_res = microgpt_classical.train_and_evaluate()
    t_classical = time.perf_counter() - t0
    
    # 2. Run Hybrid Quantum microGPT (training on clean simulator, evaluating on noisy)
    t1 = time.perf_counter()
    quantum_res = microgpt_quantum.train_and_evaluate(use_noisy_eval=True)
    t_quantum = time.perf_counter() - t1
    
    print("\n" + "=" * 90)
    print("              ABLATION STUDY: microGPT vs Hybrid Quantum microGPT")
    print("=" * 90)
    print(f"{'Metric':<30} | {'Classical microGPT':<20} | {'Quantum microGPT (Clean)':<25} | {'Quantum (Noisy Eval)':<22}")
    print("-" * 90)
    
    # Extract metrics
    c_params = classical_res["total_params"]
    q_params = quantum_res["total_params"]
    vqc_params = quantum_res["quantum_params"]
    
    c_init_loss = classical_res["losses"][0]
    q_init_loss = quantum_res["losses"][0]
    
    c_final_loss = classical_res["losses"][-1]
    q_final_loss = quantum_res["losses"][-1]
    q_noisy_val_loss = quantum_res["noisy_val_loss"]
    
    c_reduction = c_init_loss - c_final_loss
    q_reduction = q_init_loss - q_final_loss
    
    c_samples = ", ".join(classical_res["inference_samples"][:3]) + "..."
    q_samples = ", ".join(quantum_res["inference_samples"][:3]) + "..."
    
    c_time = classical_res["training_time"]
    q_time = quantum_res["training_time"]
    
    print(f"{'Total Parameters':<30} | {c_params:<20} | {q_params:<25} | {q_params:<22}")
    print(f"{'Quantum Parameters':<30} | {'N/A':<20} | {vqc_params:<25} | {vqc_params:<22}")
    print(f"{'Initial Loss (step 1)':<30} | {c_init_loss:<20.4f} | {q_init_loss:<25.4f} | {q_init_loss:<22.4f}")
    print(f"{'Final Loss (step 1000)':<30} | {c_final_loss:<20.4f} | {q_final_loss:<25.4f} | {q_noisy_val_loss:<22.4f}" if isinstance(q_noisy_val_loss, float) else f"{'Final Loss (step 1000)':<30} | {c_final_loss:<20.4f} | {q_final_loss:<25.4f} | {q_noisy_val_loss:<22}")
    print(f"{'Loss Reduction':<30} | {c_reduction:<20.4f} | {q_reduction:<25.4f} | {'N/A':<22}")
    print(f"{'Training Time (seconds)':<30} | {c_time:<20.1f} | {q_time:<25.1f} | {'N/A':<22}")
    print(f"{'Inference Samples (temp=0.5)':<30} | {c_samples:<20} | {q_samples:<25} | {'N/A':<22}")
    print("=" * 90)
    
    # 3. Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(classical_res["losses"], label="Classical microGPT", color="royalblue", alpha=0.8)
    plt.plot(quantum_res["losses"], label="Hybrid Quantum microGPT", color="darkorange", alpha=0.8)
    plt.xlabel("Training Step")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss Comparison: Classical vs Hybrid Quantum microGPT")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Save the plot
    output_plot_path = "loss_comparison.png"
    plt.savefig(output_plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Loss comparison plot saved to {output_plot_path}")

if __name__ == "__main__":
    run_ablation_study()

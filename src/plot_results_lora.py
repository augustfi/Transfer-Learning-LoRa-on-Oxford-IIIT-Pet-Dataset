import json
import matplotlib.pyplot as plt

modes = ["lora", "linear_probe", "full_finetune"]
results = {}
for m in modes:
    with open(f"results_{m}.json") as f:
        results[m] = json.load(f)

# Plot 1 : val accuracy comparison
plt.figure()
for m in modes:
    epochs = range(1, len(results[m]["val_acc"]) + 1)
    plt.plot(epochs, results[m]["val_acc"], label=m)
plt.xlabel("epoch")
plt.ylabel("val accuracy")
plt.title("Validation accuracy: LoRA vs linear probe vs full FT")
plt.legend()
plt.savefig("comparison_val_acc.png")
plt.close()

# Plot 2 : time per epoch
plt.figure()
for m in modes:
    epochs = range(1, len(results[m]["epoch_times"]) + 1)
    plt.plot(epochs, results[m]["epoch_times"], label=m)
plt.xlabel("epoch")
plt.ylabel("time (s)")
plt.title("Time per epoch")
plt.legend()
plt.savefig("comparison_time.png")
plt.close()

# Plot 3 : peak memory
plt.figure()
for m in modes:
    epochs = range(1, len(results[m]["epoch_peak_memory_gb"]) + 1)
    plt.plot(epochs, results[m]["epoch_peak_memory_gb"], label=m)
plt.xlabel("epoch")
plt.ylabel("peak GPU memory (GB)")
plt.title("Peak GPU memory per epoch")
plt.legend()
plt.savefig("comparison_memory.png")
plt.close()

# Tableau récap
print(f"\n{'mode':<20} {'test_acc':<12} {'params':<15} {'avg_time(s)':<15} {'peak_mem(GB)':<15}")
for m in modes:
    r = results[m]
    avg_time = sum(r["epoch_times"]) / len(r["epoch_times"])
    max_mem = max(r["epoch_peak_memory_gb"]) if r["epoch_peak_memory_gb"] else 0
    print(f"{m:<20} {r['test_acc']:<12.4f} {r['trainable_count']:<15,} {avg_time:<15.2f} {max_mem:<15.3f}")